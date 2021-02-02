import numpy as np
import glob
import multiprocessing
import os

import tensorflow as tf
from tf_model import load_one_file

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="cand")
    parser.add_argument("--datapath", type=str, required=True, help="Input data path")
    parser.add_argument("--num-files-per-tfr", type=int, default=100, help="Number of pickle files to merge to one TFRecord file")
    args = parser.parse_args()
    return args

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#https://stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _parse_tfr_element(element):
    parse_dic = {
        'X': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
        'w': tf.io.FixedLenFeature([], tf.string),
        #'dm_row': tf.io.FixedLenFeature([], tf.string),
        #'dm_col': tf.io.FixedLenFeature([], tf.string),
        #'dm_data': tf.io.FixedLenFeature([], tf.string),
    }
    example_message = tf.io.parse_single_example(element, parse_dic)

    X = example_message['X']
    arr_X = tf.io.parse_tensor(X, out_type=tf.float32)
    y = example_message['y']
    arr_y = tf.io.parse_tensor(y, out_type=tf.float32)
    w = example_message['w']
    arr_w = tf.io.parse_tensor(w, out_type=tf.float32)
    
    #dm_row = example_message['dm_row']
    #arr_dm_row = tf.io.parse_tensor(dm_row, out_type=tf.int64)
    #dm_col = example_message['dm_col']
    #arr_dm_col = tf.io.parse_tensor(dm_col, out_type=tf.int64)
    #dm_data = example_message['dm_data']
    #arr_dm_data = tf.io.parse_tensor(dm_data, out_type=tf.float32)

    #https://github.com/tensorflow/tensorflow/issues/24520#issuecomment-577325475
    arr_X.set_shape(tf.TensorShape((None, 15)))
    arr_y.set_shape(tf.TensorShape((None, 5)))
    arr_w.set_shape(tf.TensorShape((None, )))
    #inds = tf.stack([arr_dm_row, arr_dm_col], axis=-1)
    #dm_sparse = tf.SparseTensor(values=arr_dm_data, indices=inds, dense_shape=[tf.shape(arr_X)[0], tf.shape(arr_X)[0]])

    return arr_X, arr_y, arr_w

def serialize_X_y_w(writer, X, y, w):
    feature = {
        'X': _bytes_feature(tf.io.serialize_tensor(X)),
        'y': _bytes_feature(tf.io.serialize_tensor(y)),
        'w': _bytes_feature(tf.io.serialize_tensor(w)),
        #'dm_row': _bytes_feature(tf.io.serialize_tensor(np.array(dm.row, np.int64))),
        #'dm_col': _bytes_feature(tf.io.serialize_tensor(np.array(dm.col, np.int64))),
        #'dm_data': _bytes_feature(tf.io.serialize_tensor(dm.data)),
    }
    sample = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(sample.SerializeToString())

def serialize_chunk(args):
    path, files, ichunk, target = args
    print(path, len(files), ichunk, target)
    out_filename = os.path.join(path, "chunk_{}.tfrecords".format(ichunk))
    writer = tf.io.TFRecordWriter(out_filename)
    Xs = []
    ys = []
    ws = []
    dms = []

    for fi in files:
        print(fi)
        X, y, ycand = load_one_file(fi)

        Xs += X
        if target == "cand":
            ys += ycand
        elif target == "gen":
            ys += y
        else:
            raise Exception("Unknown target")

    #set weights for each sample to be equal to the number of samples of this type
    #in the training script, this can be used to compute either inverse or class-balanced weights
    uniq_vals, uniq_counts = np.unique(np.concatenate([y[:, 0] for y in ys]), return_counts=True)
    for i in range(len(ys)):
        w = np.ones(len(ys[i]), dtype=np.float32)
        for uv, uc in zip(uniq_vals, uniq_counts):
            w[ys[i][:, 0]==uv] = uc
        ws += [w]

    for X, y, w in zip(Xs, ys, ws):
        #print("serializing", X.shape, y.shape, w.shape)
        serialize_X_y_w(writer, X, y, w)

    writer.close()

if __name__ == "__main__":
    args = parse_args()
    #tf.config.experimental_run_functions_eagerly(True)

    datapath = args.datapath

    filelist = sorted(glob.glob("{}/raw/*.pkl".format(datapath)))
    print("found {} files".format(len(filelist)))
    assert(len(filelist) > 0)
    #means, stds = extract_means_stds(filelist)
    outpath = "{}/tfr/{}".format(datapath, args.target)

    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    pars = []
    for ichunk, files in enumerate(chunks(filelist, args.num_files_per_tfr)):
        pars += [(outpath, files, ichunk, args.target)]
    assert(len(pars) > 0)
    #serialize_chunk(pars[0])
    #pool = multiprocessing.Pool(20)
    for par in pars:
        serialize_chunk(par)

    #Load and test the dataset 
    tfr_dataset = tf.data.TFRecordDataset(glob.glob(outpath + "/*.tfrecords"))
    dataset = tfr_dataset.map(_parse_tfr_element)
    num_ev = 0
    num_particles = 0
    for X, y, w in dataset:
        num_ev += 1
        num_particles += len(X)
    assert(num_ev > 0)
    print("Created TFRecords dataset in {} with {} events, {} particles".format(
        datapath, num_ev, num_particles))
