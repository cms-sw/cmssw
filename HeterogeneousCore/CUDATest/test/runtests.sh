#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

TEST_DIR=$CMSSW_BASE/src/HeterogeneousCore/CUDATest/test

if [ "x$#" != "x1" ]; then
    die "Need exactly 1 argument ('cpu', 'gpu'), got $#" 1
fi
if [ "x$1" = "xgpu" ]; then
    TARGET=gpu
elif [ "x$1" = "xcpu" ]; then
    # In non-_GPU_ IBs, if CUDA is enabled, run the GPU-targeted tests
    cudaIsEnabled
    CUDA_ENABLED=$?
    if [ "x${CUDA_ENABLED}" == "x0" ]; then
        TARGET=gpu
    else
        TARGET=cpu
    fi
else
    die "Argument needs to be 'cpu' or 'gpu', got $1" 1
fi

echo "*************************************************"
echo "CUDA producer configuration with SwitchProducer, automatic"
cmsRun ${TEST_DIR}/testCUDASwitch_cfg.py --silent || die "cmsRun testCUDASwitch_cfg.py --silent" $?

echo "*************************************************"
echo "CUDA producer configuration with SwitchProducer, force CPU"
cmsRun ${TEST_DIR}/testCUDASwitch_cfg.py --silent --accelerator="cpu" || die "cmsRun testCUDASwitch_cfg.py --silent --accelerator=\"\"" $?

if [ "x${TARGET}" == "xgpu" ]; then
    echo "*************************************************"
    echo "CUDA producer configuration with SwitchProducer, force GPU"
    cmsRun ${TEST_DIR}/testCUDASwitch_cfg.py --silent --accelerator="gpu-nvidia" || die "cmsRun testCUDASwitch_cfg.py --silent --accelerator=gpu-nvidia" $?
elif [ "x${TARGET}" == "xcpu" ]; then
    echo "*************************************************"
    echo "CUDA producer configuration with SwitchProducer, force GPU, should fail"
    cmsRun -j testCUDATest_jobreport.xml ${TEST_DIR}/testCUDASwitch_cfg.py --silent --accelerator="gpu-nvidia" && die "cmsRun testCUDASwitch_cfg.py --silent --accelerator=gpu-nvidia did not fail" 1
    EXIT_CODE=$(edmFjrDump --exitCode testCUDATest_jobreport.xml)
    if [ "x${EXIT_CODE}" != "x8035" ]; then
        echo "Test (that was expected to fail) reported exit code ${EXIT_CODE} instead of expected 8035"
    fi
fi
