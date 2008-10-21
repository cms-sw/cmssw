#!/bin/bash

test=testRunMerge

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD1_cfg.py || die "cmsRun ${test}PROD1_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD2_cfg.py || die "cmsRun ${test}PROD2_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD3_cfg.py || die "cmsRun ${test}PROD3_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD4_cfg.py || die "cmsRun ${test}PROD4_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD5_cfg.py || die "cmsRun ${test}PROD5_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE_cfg.py || die "cmsRun ${test}MERGE_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}TEST_cfg.py || die "cmsRun ${test}TEST_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}TEST1_cfg.py || die "cmsRun ${test}TEST1_cfg.py" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}COPY_cfg.py || die "cmsRun ${test}COPY_cfg.py" $?

popd

exit 0
