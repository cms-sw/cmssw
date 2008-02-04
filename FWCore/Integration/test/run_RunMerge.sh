#!/bin/bash

test=testRunMerge

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD1.cfg || die "cmsRun ${test}PROD1.cfg" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD2.cfg || die "cmsRun ${test}PROD2.cfg" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD3.cfg || die "cmsRun ${test}PROD3.cfg" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE.cfg || die "cmsRun ${test}MERGE.cfg" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}TEST.cfg || die "cmsRun ${test}TEST.cfg" $?

popd

exit 0
