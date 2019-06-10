#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  echo "*************************************************"
  echo "CUDA producer configuration with SwitchProducer"
  cmsRun ${LOCAL_TEST_DIR}/testCUDASwitch_cfg.py || die "cmsRun testCUDASwitch_cfg.py 1" $?

popd
