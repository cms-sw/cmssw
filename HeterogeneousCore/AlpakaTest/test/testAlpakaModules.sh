#!/bin/bash

function die { echo Failed $1: status $2 ; exit $2 ; }

TEST_DIR=${LOCALTOP}/src/HeterogeneousCore/AlpakaTest/test

if [ "$#" != "1" ]; then
    die "Need exactly 1 argument ('cpu', 'cuda'), got $#" 1
fi
if [ "$1" = "cuda" ]; then
    TARGET=cuda
elif [ "$1" = "cpu" ]; then
    # In non-_GPU_ IBs, if CUDA is enabled, run the GPU-targeted tests
    cudaIsEnabled
    CUDA_ENABLED=$?
    if [ "${CUDA_ENABLED}" = "0" ]; then
        TARGET=cuda
    else
        TARGET=cpu
    fi
else
    die "Argument needs to be 'cpu' or 'cpu', got $1" 1
fi

echo "cmsRun testAlpakaModules_cfg.py"
cmsRun ${TEST_DIR}/testAlpakaModules_cfg.py || die "cmsRun testAlpakaModules_cfg.py" $?

if [ "x${TARGET}" == "xcuda" ]; then
    echo "cmsRun testAlpakaModules_cfg.py --cuda"
    cmsRun ${TEST_DIR}/testAlpakaModules_cfg.py -- --cuda || die "cmsRun testAlpakaModules_cfg.py --cuda" $?
fi
