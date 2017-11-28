#!/bin/bash

test=testParentageWithStreamerIO

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  echo ${test}_1_cfg.py
  cmsRun -p ${LOCAL_TEST_DIR}/${test}_1_cfg.py || die "cmsRun ${test}_1_cfg.py" $?

  echo ${test}_2_cfg.py
  cmsRun -p ${LOCAL_TEST_DIR}/${test}_2_cfg.py || die "cmsRun ${test}_2_cfg.py" $?

  echo ${test}_3_cfg.py
  cmsRun -p ${LOCAL_TEST_DIR}/${test}_3_cfg.py || die "cmsRun ${test}_3_cfg.py" $?

popd

exit 0
