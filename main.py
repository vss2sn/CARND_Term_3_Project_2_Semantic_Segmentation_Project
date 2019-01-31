#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

EPOCHS = 40
BATCH_SIZE = 10

LEARNING_RATE = 0.0001
DROPOUT = 0.5

WEIGHTS_INIT_SD = 0.01
WEIGHTS_REGU_L2 = 1e-3

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_layer, keep_prob, layer_3, layer_4, layer_7
tests.test_load_vgg(load_vgg, tf)

# Returns the output from a 1x1 convolution layer
def conv_1x1(layer, name, num_classes):
    return tf.layers.conv2d(inputs = layer, filters = num_classes, kernel_size = (1,1), strides = (1,1), padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev=WEIGHTS_INIT_SD), kernel_regularizer= tf.contrib.layers.l2_regularizer(WEIGHTS_REGU_L2), name = name)
    #return tf.layers.conv2d(inputs = layer, filters = num_classes, kernel_size = (1,1), strides = (1,1), padding = 'same', name = name)

# Return the output from transpose convolution
def upsample(layer, k, s, name, num_classes):
    return tf.layers.conv2d_transpose(inputs = layer, filters = num_classes, kernel_size = (k, k), strides = (s, s), padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev=WEIGHTS_INIT_SD), kernel_regularizer= tf.contrib.layers.l2_regularizer(WEIGHTS_REGU_L2), name = name)
    #return tf.layers.conv2d_transpose(inputs = layer, filters = num_classes, kernel_size = (k, k), strides = (s, s), padding = 'same', name = name)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    conv_1x1_7 = conv_1x1(layer = vgg_layer7_out, name = "conv_1x1_7", num_classes = num_classes)
    conv_1x1_4 = conv_1x1(layer = vgg_layer4_out, name = "conv_1x1_4", num_classes = num_classes)
    conv_1x1_3 = conv_1x1(layer = vgg_layer3_out, name = "conv_1x1_3", num_classes = num_classes)

    decoder_1 = upsample(layer = conv_1x1_7, k = 4, s = 2, name = "decoder_1", num_classes = num_classes)
    decoder_1_skip = tf.add(decoder_1, conv_1x1_4, name = "decoder_1_skip")
    decoder_2 = upsample(layer = decoder_1_skip, k = 4, s = 2, name = "decoder_2", num_classes = num_classes)
    decoder_2_skip = tf.add(decoder_2, conv_1x1_3, name = "decoder_2_skip")
    output = upsample(layer = decoder_2_skip, k = 16, s = 8, name = "output", num_classes = num_classes)

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = tf.reduce_sum(reg_loss)
    
    #loss = unreg_loss
    # cross_entropy_loss = tf.add(cross_entropy_loss, reg_loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss, reg_loss
#tests.test_optimize(optimize)


def train_nn(reg_loss, sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print("Training started. \n Number of epoches: " + str(EPOCHS) + "\n Batch size = " + str(BATCH_SIZE) + "\n")
    for epoch in range(EPOCHS):
        print('Starting epoch ' + str(epoch+1) + '/' + str(EPOCHS))
        loss_main = []
        i = 0
        for images, labels in get_batches_fn(BATCH_SIZE):
            i+=1
            feed_dict = {input_image:images, correct_label: labels, keep_prob: DROPOUT, learning_rate: LEARNING_RATE}
            _, loss, new_reg_loss = sess.run([train_op, cross_entropy_loss, reg_loss], feed_dict = feed_dict)
            loss_main.append(loss)
	    # Following line added to show how much regularization contributes to loss
            print("Reg loss for batch " + str(i) + " : " + str(new_reg_loss))
            print("Loss for batch " + str(i) + " : " + str(loss))
        training_loss = sum(loss_main)/len(loss_main)
        print("\nLoss for epoch " + str(epoch + 1) + "/" + str(EPOCHS) + " : " + str(training_loss) + "\n")
    print("Training complete.")
#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = "./data"
    runs_dir = "./runs"
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate = tf.placeholder(tf.float32)
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss, reg_loss = optimize(output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_nn(reg_loss, sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

