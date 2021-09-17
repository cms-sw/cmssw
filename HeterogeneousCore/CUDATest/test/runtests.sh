#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

TEST_DIR=src/HeterogeneousCore/CUDATest/test

echo "*************************************************"
echo "CUDA producer configuration with SwitchProducer"
cmsRun ${TEST_DIR}/testCUDASwitch_cfg.py || die "cmsRun testCUDASwitch_cfg.py 1" $?
