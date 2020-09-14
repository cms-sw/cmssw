#!/bin/bash

# borrowed from https://github.com/NVIDIA/triton-inference-server/tree/master/docs/examples

TRITON_VERSION=$(scram tool info triton-inference-server | grep "Version : " | cut -d' ' -f3)

TEST_DIR=`pwd`

MODEL_DIR=${TEST_DIR}/../data/models/resnet50_netdef
cd $TEST_DIR
mkdir -p $MODEL_DIR
cd $MODEL_DIR

curl -O -L https://github.com/NVIDIA/triton-inference-server/raw/v${TRITON_VERSION}/docs/examples/model_repository/resnet50_netdef/config.pbtxt
curl -O -L https://github.com/NVIDIA/triton-inference-server/raw/v${TRITON_VERSION}/docs/examples/model_repository/resnet50_netdef/resnet50_labels.txt

mkdir -p 1

curl -o 1/model.netdef http://download.caffe2.ai.s3.amazonaws.com/models/resnet50/predict_net.pb
curl -o 1/init_model.netdef http://download.caffe2.ai.s3.amazonaws.com/models/resnet50/init_net.pb

GAT_DIR=${TEST_DIR}/../data/models/gat_test
cd $TEST_DIR
mkdir -p $GAT_DIR
cd $GAT_DIR

curl -O -L https://github.com/lgray/triton-torchgeo-gat-example/raw/cmssw_20.06-v1-py3/artifacts/models/gat_test/config.pbtxt
mkdir -p 1
curl -o 1/model.pt -L https://github.com/lgray/triton-torchgeo-gat-example/raw/cmssw_20.06-v1-py3/artifacts/models/gat_test/1/model.pt

TORCH_DIR=${TEST_DIR}/../data/lib/
cd $TEST_DIR
mkdir -p $TORCH_DIR
cd $TORCH_DIR

for lib in libtorchcluster.so libtorchscatter.so libtorchsparse.so libtorchsplineconv.so; do
  curl -O -L https://github.com/lgray/triton-torchgeo-gat-example/raw/cmssw_20.06-v1-py3/artifacts/lib/$lib
done
