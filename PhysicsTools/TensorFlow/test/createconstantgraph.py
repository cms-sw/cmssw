# coding: utf-8

"""
Test script to create a simple graph for testing purposes at bin/data and save it with all
variables converted to constants to reduce its memory footprint.

https://www.tensorflow.org/api_docs/python/tf/graph_util/convert_variables_to_constants
"""

import os
import sys

import cmsml


# get tensorflow and work with the v1 compatibility layer
tf, tf1, tf_version = cmsml.tensorflow.import_tf()
tf = tf1
tf.disable_eager_execution()

# prepare the datadir
if len(sys.argv) >= 2:
    datadir = sys.argv[1]
else:
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

# create the graph
x_ = tf.placeholder(tf.float32, [None, 10], name="input")
scale_ = tf.placeholder(tf.float32, name="scale")

W = tf.Variable(tf.ones([10, 1]))
b = tf.Variable(tf.ones([1]))
h = tf.add(tf.matmul(x_, W), b)
y = tf.multiply(h, scale_, name="output")


# Setup the script to run on CPU only
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

print(sess.run(y, feed_dict={scale_: 1.0, x_: [range(10)]})[0][0])

# write it
graph_path = os.path.join(datadir, "constantgraph.pb")
outputs = ["output"]
cmsml.tensorflow.save_graph(graph_path, sess, output_names=outputs, variables_to_constants=True)
