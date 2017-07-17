#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest1_cfg.py || die "cmsRun ThinningTest1_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest2_cfg.py || die "cmsRun ThinningTest2_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest3_cfg.py || die "cmsRun ThinningTest3_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest4_cfg.py || die "cmsRun ThinningTest4_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTest_dropOnInput_cfg.py || die "cmsRun ThinningTest_dropOnInput_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTestSubProcess_cfg.py || die "cmsRun ThinningTestSubProcess_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTestSubProcessRead_cfg.py || die "cmsRun ThinningTestSubProcessRead_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/ThinningTestStreamerIn_cfg.py || die "cmsRun ThinningTestStreamerIn_cfg.py" $?

popd

exit 0
