#!/bin/bash

test=testRandomService

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun -p ${LOCAL_TEST_DIR}/${test}1.cfg || die "cmsRun ${test}1.cfg" $?
  mv ${test}.txt ${test}1.txt 
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/${test}1.txt ${LOCAL_TMP_DIR}/${test}1.txt || die "comparing ${test}1.txt" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}2.cfg || die "cmsRun ${test}2.cfg" $?
  mv ${test}.txt ${test}2.txt 
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/${test}2.txt ${LOCAL_TMP_DIR}/${test}2.txt || die "comparing ${test}2.txt" $?

popd

exit 0
