#!/bin/bash

test=ref_merge_

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
  echo ${test}prod1 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}prod1_cfg.py || die "cmsRun ${test}prod1_cfg.py" $?

  echo ${test}prod2 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}prod2_cfg.py || die "cmsRun ${test}prod2_cfg.py" $?

  echo ${test}MERGE------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}cfg.py || die "cmsRun ${test}cfg.py" $?

  echo ${test}test------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}test_cfg.py || die "cmsRun ${test}test_cfg.py" $?

  echo ${test}test------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}subprocess_cfg.py || die "cmsRun ${test}subprocess_cfg.py" $?

exit 0
