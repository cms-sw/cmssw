# coding: utf-8

"""
Test script to create a simple graph for testing purposes at bin/data and save it with all
variables converted to constants to reduce its memory footprint.

https://www.tensorflow.org/api_docs/python/tf/graph_util/convert_variables_to_constants
"""


import os
import sys
import tensorflow as tf

from PhysicsTools.TensorFlow.tools import TF2, write_constant_graph


# go into v1 compatibility mode
if TF2:
    tf = tf.compat.v1
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

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(y, feed_dict={scale_: 1.0, x_: [range(10)]})[0][0])

# write it
outputs = ["output"]
write_constant_graph(sess, ["output"], os.path.join(datadir, "constantgraph.pb"))
