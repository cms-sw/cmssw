#!/bin/bash

test=testDropOnInput

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  echo "testDropOnInput1_1"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}1_1_cfg.py || die "cmsRun ${test}1_1_cfg.py" $?

  echo "testDropOnInput1_2"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}1_2_cfg.py || die "cmsRun ${test}1_2_cfg.py" $?

  echo "testDropOnInput2"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}2_cfg.py || die "cmsRun ${test}2_cfg.py" $?

  echo "testDropOnInput3"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}3_cfg.py || die "cmsRun ${test}3_cfg.py" $?

  echo "testDropOnInputRead2"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}Read2_cfg.py || die "cmsRun ${test}Read2_cfg.py" $?

  echo "testDropOnInputRead2001"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}Read2001_cfg.py || die "cmsRun ${test}Read2001_cfg.py" $?

  echo "testDropOnInputRead3"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}Read3_cfg.py || die "cmsRun ${test}Read3_cfg.py" $?

  echo "testDropOnInputSubProcess_cfg.py"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}SubProcess_cfg.py || die "cmsRun ${test}SubProcess_cfg.py" $?

  echo "testDropOnInputReadSubProcess_cfg.py"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}ReadSubProcess_cfg.py || die "cmsRun ${test}ReadSubProcess_cfg.py" $?

popd

exit 0

