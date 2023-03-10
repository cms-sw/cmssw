#!/bin/bash

test=testRunMerge

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}
  echo ${test}PROD100 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD100_cfg.py || die "cmsRun ${test}PROD100_cfg.py" $?

  echo ${test}PROD101 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD101_cfg.py || die "cmsRun ${test}PROD101_cfg.py" $?

  echo ${test}PROD102 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD102_cfg.py || die "cmsRun ${test}PROD102_cfg.py" $?

  echo ${test}SPLIT100 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}SPLIT100_cfg.py || die "cmsRun ${test}SPLIT100_cfg.py" $?

  echo ${test}SPLIT101 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}SPLIT101_cfg.py || die "cmsRun ${test}SPLIT101_cfg.py" $?

  echo ${test}SPLIT102 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}SPLIT102_cfg.py || die "cmsRun ${test}SPLIT102_cfg.py" $?

  echo ${test}SPLIT103 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}SPLIT103_cfg.py || die "cmsRun ${test}SPLIT103_cfg.py" $?

  echo ${test}MERGE100 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE100_cfg.py || die "cmsRun ${test}MERGE100_cfg.py" $?

  echo ${test}MERGE101 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE101_cfg.py || die "cmsRun ${test}MERGE101_cfg.py" $?

  echo ${test}MERGE102 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE102_cfg.py || die "cmsRun ${test}MERGE102_cfg.py" $?

  echo ${test}TEST100 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}TEST100_cfg.py || die "cmsRun ${test}TEST100_cfg.py" $?

  echo ${test}TEST101 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}TEST101_cfg.py || die "cmsRun ${test}TEST101_cfg.py" $?

  echo ${test}COPY100 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}COPY100_cfg.py || die "cmsRun ${test}COPY100_cfg.py" $?

  echo ${test}COPY101 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}COPY101_cfg.py || die "cmsRun ${test}COPY101_cfg.py" $?

exit 0
