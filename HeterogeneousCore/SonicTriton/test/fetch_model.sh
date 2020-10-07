#!/bin/bash

# borrowed from https://github.com/triton-inference-server/server/tree/master/docs/examples

TRITON_REPO="https://github.com/triton-inference-server/server"
TRITON_VERSION=$(scram tool info triton-inference-server | grep "Version : " | cut -d' ' -f3 | cut -d'-' -f1)

TEST_DIR=`pwd`

MODEL_DIR=${TEST_DIR}/../data/models/resnet50_netdef
cd $TEST_DIR
mkdir -p $MODEL_DIR
cd $MODEL_DIR

curl -O -L ${TRITON_REPO}/raw/v${TRITON_VERSION}/docs/examples/model_repository/resnet50_netdef/config.pbtxt
curl -O -L ${TRITON_REPO}/raw/v${TRITON_VERSION}/docs/examples/model_repository/resnet50_netdef/resnet50_labels.txt

mkdir -p 1

curl -o 1/model.netdef http://download.caffe2.ai.s3.amazonaws.com/models/resnet50/predict_net.pb
curl -o 1/init_model.netdef http://download.caffe2.ai.s3.amazonaws.com/models/resnet50/init_net.pb

GAT_DIR=${TEST_DIR}/../data/models/gat_test
cd $TEST_DIR
mkdir -p $GAT_DIR
cd $GAT_DIR

cat << EOF > config.pbtxt
name: "gat_test"
platform: "pytorch_libtorch"
max_batch_size: 0
input [
  {
    name: "x__0"
    data_type: TYPE_FP32
    dims: [ -1, 1433 ]
  },
  {
    name: "edgeindex__1"
    data_type: TYPE_INT64
    dims: [ 2, -1 ]
  }
]
output [
  {
    name: "logits__0"
    data_type: TYPE_FP32
    dims: [ -1, 7 ]
  }
]
EOF

mkdir -p 1
cp /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fastml/triton-torchgeo:20.09-py3-geometric/torch_geometric/examples/model.pt 1/model.pt
