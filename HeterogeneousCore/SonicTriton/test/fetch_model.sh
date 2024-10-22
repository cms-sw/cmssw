#!/bin/bash

# borrowed from https://github.com/triton-inference-server/server/tree/master/docs/examples

TRITON_REPO="https://github.com/triton-inference-server/server"
TRITON_VERSION=$(scram tool info triton-inference-client | grep "Version : " | cut -d' ' -f3 | cut -d'-' -f1)

TEST_DIR=`pwd`

get_model(){
	MODEL_NAME="$1"

	MODEL_DIR=${TEST_DIR}/../data/models/${MODEL_NAME}
	cd $TEST_DIR
	mkdir -p $MODEL_DIR
	cd $MODEL_DIR

	if [[ "$MODEL_NAME" == inception_graphdef ]]; then
		FNAME=inception_v3_2016_08_28_frozen.pb.tar.gz
		mkdir -p 1
		mkdir -p tmp
		wget -O tmp/${FNAME} https://storage.googleapis.com/download.tensorflow.org/models/${FNAME}
		(cd tmp && tar -xzf ${FNAME})
		mv tmp/inception_v3_2016_08_28_frozen.pb 1/model.graphdef
		rm -rf tmp
	elif [[ "$MODEL_NAME" == densenet_onnx ]]; then
		mkdir -p 1
		wget -O 1/model.onnx https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx
	fi

	curl -O -L ${TRITON_REPO}/raw/v${TRITON_VERSION}/docs/examples/model_repository/${MODEL_NAME}/config.pbtxt
	curl -O -L ${TRITON_REPO}/raw/v${TRITON_VERSION}/docs/examples/model_repository/${MODEL_NAME}/$(echo $MODEL_NAME | cut -d'_' -f1)_labels.txt
}

get_model inception_graphdef
get_model densenet_onnx
