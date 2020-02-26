# coding: utf-8

"""
TensorFlow tools and helpers.
"""


__all__ = ["TF1", "TF2", "read_constant_graph", "write_constant_graph", "visualize_graph"]


import os
import sys
import shutil
import tempfile
import signal
import subprocess

import six
import tensorflow as tf


# version flags
TF1 = tf.__version__.startswith("1.")
TF2 = tf.__version__.startswith("2.")

# complain when the version is not yet covered
if not TF1 and not TF2:
    raise NotImplementedError("TensorFlow version {} is not yet supported".format(tf.__version__))

# keep a reference to the v1 API as long as v2 provides compatibility
tf1 = None
if TF1:
    tf1 = tf
elif getattr(tf, "compat", None) and getattr(tf.compat, "v1"):
    tf1 = tf.compat.v1


def read_constant_graph(graph_path, create_session=None, as_text=None):
    """
    Reads a saved TensorFlow graph from *graph_path* and returns it. When *create_session* is
    *True*, a session object (compatible with the v1 API) is created and returned as well as the
    second value of a 2-tuple. The default value of *create_session* is *True* when TensorFlow v1
    is detected, and *False* otherwise. When *as_text* is *True*, or *None* and the file extension
    is ``".pbtxt"`` or ``".pb.txt"``, the content of the file at *graph_path* is expected to be a
    human-readable text file. Otherwise, it is expected to be a binary protobuf file. Example:

    .. code-block:: python

        graph = read_constant_graph("path/to/model.pb", create_session=False)

        graph, session = read_constant_graph("path/to/model.pb", create_session=True)
    """
    if as_text is None:
        as_text = graph_path.endswith((".pbtxt", ".pb.txt"))

    graph = tf.Graph()
    with graph.as_default():
        graph_def = graph.as_graph_def()

        if as_text:
            # use a simple pb reader to load the file into graph_def
            from google.protobuf import text_format
            with open(graph_path, "r") as f:
                text_format.Merge(f.read(), graph_def)

        else:
            # use the gfile api depending on the TF version
            if TF1:
                from tensorflow.python.platform import gfile
                with gfile.FastGFile(graph_path, "rb") as f:
                    graph_def.ParseFromString(f.read())
            else:
                with tf.io.gfile.GFile(graph_path, "rb") as f:
                    graph_def.ParseFromString(f.read())

        # import the graph_def (pb object) into the actual graph
        tf.import_graph_def(graph_def, name="")

    # determine the create_session default
    if create_session is None:
        create_session = TF1

    if create_session:
        if not tf1:
            raise NotImplementedError("the v1 compatibility layer of TensorFlow v2 (tf.compat.v1) "
                "is required by read_constant_graph when create_session is True, but missing")
        session = tf1.Session(graph=graph)
        return graph, session
    else:
        return graph


def write_constant_graph(session, output_names, graph_path, **kwargs):
    """
    Takes a TensorFlow *session* object (compatible with the v1 API), converts its contained graph
    into a simpler version with variables translated into constant tensors, and saves it to a pb
    file defined by *graph_path*. *output_numes* must be a list of names of output tensors to save.
    In turn, TensorFlow internally determines which subgraph(s) to convert and save. All *kwargs*
    are forwarded to :py:func:`tf.compat.v1.train.write_graph`. Intermediate output directories are
    created, the output file is removed when already existing, and the absolute and normalized
    output path is returned.

    .. note::

        When used with TensorFlow v2, this function requires the v1 API compatibility layer. When
        :py:attr:`tf.compat.v1` is not available, a *NotImplementedError* is raised.
    """
    # complain when the v1 compatibility layer is not existing
    if not tf1:
        raise NotImplementedError("the v1 compatibility layer of TensorFlow v2 (tf.compat.v1) is "
            "required by write_constant_graph, but missing")

    # convert the graph
    constant_graph = tf1.graph_util.convert_variables_to_constants(session,
        session.graph.as_graph_def(), output_names)

    # prepare the output path
    graph_path = os.path.normpath(os.path.abspath(graph_path))
    graph_dir, graph_name = os.path.split(graph_path)
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    if os.path.exists(graph_path):
        os.remove(graph_path)

    # write the graph
    kwargs.setdefault("as_text", False)
    tf1.train.write_graph(constant_graph, graph_dir, graph_name, **kwargs)

    return graph_path


def visualize_graph(graph, log_dir=None, start_tensorboard=False, tensorboard_args="", **kwargs):
    """
    Visualizes a TensorFlow *graph* by adding it to a ``tf.summary.FileWriter``. *graph* can be
    either a graph object or a path to a pb file. In the latter case, :py:func:`read_constant_graph`
    is used and all *kwargs* are forwarded. The file writer object is instantiated with a *log_dir*
    which, when empty, defaults to a temporary directory. This is especially usefull when
    *start_tensorboard* is *True*, in which case a subprocesses is started to run a *tensorboard*
    instance with additional arguments given as a string *tensorboard_args*. The subprocess is
    terminated on keyboard interrupt.

    .. note::

        When used with TensorFlow v2, this function requires the v1 API compatibility layer. When
        :py:attr:`tf.compat.v1` is not available, a *NotImplementedError* is raised.
    """
    # complain when the v1 compatibility layer is not existing
    if not tf1:
        raise NotImplementedError("the v1 compatibility layer of TensorFlow v2 (tf.compat.v1) is "
            "required by visualize_graph, but missing")

    # prepare the log_dir
    is_tmp = not log_dir
    if is_tmp:
        log_dir = tempfile.mkdtemp()
    elif not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # read the graph when a string is passed
    if isinstance(graph, six.string_types):
        graph = read_constant_graph(graph, create_session=False, **kwargs)

    # switch to non-eager mode for the FileWriter to work
    eager = getattr(tf1, "executing_eagerly", lambda: False)()
    if eager:
        tf1.disable_eager_execution()

    # write to file
    writer = tf1.summary.FileWriter(log_dir)
    writer.add_graph(graph)

    # reset the eager mode
    if eager:
        tf1.enable_eager_execution()

    # optionally start a tensorboard process
    if start_tensorboard:
        print("starting tensorboard with logdir {}".format(log_dir))
        cmd = "tensorboard --logdir '{}' {}".format(log_dir, tensorboard_args)
        p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", preexec_fn=os.setsid)
        try:
            p.communicate()
        except (Exception, KeyboardInterrupt):
            print("tensorboard terminated")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            pass

    # cleanup when log_dir is temporary
    if is_tmp:
        shutil.rmtree(log_dir)


def _test():
    """
    Internal test of the above functions based on the deepjet model.
    """
    deepjet_model = "/cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/data-RecoBTag-Combined/V01-02-01/" \
        "RecoBTag/Combined/data/DeepFlavourV03_10X_training/constant_graph.pb"
    deepjet_output = "ID_pred/Softmax"

    if not os.path.exists(deepjet_model):
        print("cannot run tests as deepjet model '{}' does not exist".format(deepjet_model))
        sys.exit(1)

    # load the graph
    read_constant_graph(deepjet_model, create_session=False)
    if tf1:
        g, s = read_constant_graph(deepjet_model, create_session=True)

    # write the graph
    if tf1:
        with tempfile.NamedTemporaryFile(suffix=".pb") as ntf:
            write_constant_graph(s, [deepjet_output], ntf.name)

    # visualize the graph
    if tf1:
        visualize_graph(g)


if __name__ == "__main__":
    _test()
