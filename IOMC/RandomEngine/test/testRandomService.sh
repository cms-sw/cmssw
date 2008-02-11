#!/bin/bash

test=testRandomService

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  echo " "
  echo "RandomNumberGeneratorService test 1"
  echo "==================================="
  cmsRun -p ${LOCAL_TEST_DIR}/${test}1.cfg || die "cmsRun ${test}1.cfg" $?
  mv ${test}.txt ${test}1.txt 
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/${test}1.txt ${LOCAL_TMP_DIR}/${test}1.txt || die "comparing ${test}1.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test 2"
  echo "==================================="
  cmsRun -p ${LOCAL_TEST_DIR}/${test}2.cfg || die "cmsRun ${test}2.cfg" $?
  mv ${test}.txt ${test}2.txt 
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/${test}2.txt ${LOCAL_TMP_DIR}/${test}2.txt || die "comparing ${test}2.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test 3"
  echo "==================================="
  cmsRun -p ${LOCAL_TEST_DIR}/${test}3.cfg || die "cmsRun ${test}3.cfg" $?
  mv ${test}.txt ${test}3.txt 
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/${test}3.txt ${LOCAL_TMP_DIR}/${test}3.txt || die "comparing ${test}3.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test 4"
  echo "==================================="
  cmsRun -p ${LOCAL_TEST_DIR}/${test}4.cfg || die "cmsRun ${test}4.cfg" $?
  mv ${test}.txt ${test}4.txt 
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/${test}4.txt ${LOCAL_TMP_DIR}/${test}4.txt || die "comparing ${test}4.txt" $?

popd

exit 0
