# -*- coding: utf-8 -*-

"""
Test script to create a large graph for testing purposes at bin/data and save it using the
SavedModel serialization format.

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md
"""


import os
import sys
import tensorflow as tf


if len(sys.argv) >= 2:
    datadir = sys.argv[1]
else:
    thisdir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

x_ = tf.placeholder(tf.float32, [None, 100], name="input")

prev_output = x_
prev_units = 100
nn = 5 * [200]

for units in nn:
    W = tf.Variable(tf.random_normal([prev_units, units]))
    b = tf.Variable(tf.random_normal([units]))
    h = tf.nn.elu(tf.matmul(prev_output, W) + b)
    prev_units = units
    prev_output = h

W_last = tf.Variable(tf.random_normal([prev_units, 10]))
b_last = tf.Variable(tf.random_normal([10]))
y  = tf.nn.softmax(tf.matmul(prev_output, W_last) + b_last, name="output")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(y, feed_dict={x_: [range(100)]})[0])

builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(datadir, "largegraph"))
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save()
