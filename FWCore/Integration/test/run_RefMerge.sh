#!/bin/bash

test=ref_merge_

function die { echo Failure $1: status $2 ; exit $2 ; }

echo LOCAL_TMP_DIR = ${LOCAL_TMP_DIR}

pushd ${LOCAL_TMP_DIR}
  echo ${test}prod1 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}prod1_cfg.py || die "cmsRun ${test}prod1_cfg.py" $?

  echo ${test}prod2 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}prod2_cfg.py || die "cmsRun ${test}prod2_cfg.py" $?

  echo ${test}MERGE------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}cfg.py || die "cmsRun ${test}cfg.py" $?

  echo ${test}test------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}test_cfg.py || die "cmsRun ${test}test_cfg.py" $?

popd

exit 0
