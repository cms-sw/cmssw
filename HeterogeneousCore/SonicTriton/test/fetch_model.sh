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

GAT_REPO="https://github.com/lgray/triton-torchgeo-gat-example/raw/cmssw_20.06-v1-py3"
GAT_DIR=${TEST_DIR}/../data/models/gat_test
cd $TEST_DIR
mkdir -p $GAT_DIR
cd $GAT_DIR

curl -O -L ${GAT_REPO}/artifacts/models/gat_test/config.pbtxt
mkdir -p 1
curl -o 1/model.pt -L ${GAT_REPO}/artifacts/models/gat_test/1/model.pt
