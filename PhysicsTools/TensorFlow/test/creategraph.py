# -*- coding: utf-8 -*-

"""
Test script to create a simple graph for testing purposes at bin/data and save it using the
SavedModel serialization format.

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md
"""


import os
import tensorflow as tf


thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(thisdir), "bin", "data")

x_ = tf.placeholder(tf.float32, [None, 10], name="input")

W = tf.Variable(tf.ones([10, 1]))
b = tf.Variable(tf.ones([1]))
y = tf.add(tf.matmul(x_, W), b, name="output")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(y, feed_dict={x_: [range(10)]})[0][0])

builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(datadir, "simplegraph"))
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save()
