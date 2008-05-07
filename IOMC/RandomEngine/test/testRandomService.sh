#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

newtest=testRandomService
oldtest=oldStyle

pushd ${LOCAL_TMP_DIR}

  echo " "
  echo "RandomNumberGeneratorService test new format 1"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${newtest}1.cfg 
  mv ${newtest}.txt ${newtest}1.txt
  diff ${LOCAL_TMP_DIR}/${newtest}1.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${newtest}1.txt || die "comparing ${newtest}1.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test new format 2"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${newtest}2.cfg 
  mv ${newtest}.txt ${newtest}2.txt
  diff ${LOCAL_TMP_DIR}/${newtest}2.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${newtest}2.txt || die "comparing ${newtest}2.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test new format 3"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${newtest}3.cfg 
  mv ${newtest}.txt ${newtest}3.txt
  diff ${LOCAL_TMP_DIR}/${newtest}3.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${newtest}3.txt || die "comparing ${newtest}3.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test new format 4"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${newtest}4.cfg 
  mv ${newtest}.txt ${newtest}4.txt
  diff ${LOCAL_TMP_DIR}/${newtest}4.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${newtest}4.txt || die "comparing ${newtest}4.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test old format 1"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${oldtest}1.cfg
  mv ${newtest}.txt ${newtest}1.txt
  diff ${LOCAL_TMP_DIR}/${newtest}1.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${newtest}1.txt || die "comparing ${newtest}1.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test old format 2"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${oldtest}2.cfg 
  mv ${newtest}.txt ${newtest}2.txt
  diff ${LOCAL_TMP_DIR}/${newtest}2.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${newtest}2.txt || die "comparing ${newtest}2.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test old format 3"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${oldtest}3.cfg 
  mv ${newtest}.txt ${newtest}3.txt
  diff ${LOCAL_TMP_DIR}/${newtest}3.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${newtest}3.txt || die "comparing ${newtest}3.txt" $?

  echo " "
  echo "RandomNumberGeneratorService test old format 4"
  echo "=============================================="
  cmsRun ${LOCAL_TEST_DIR}/${oldtest}4.cfg 
  mv ${newtest}.txt ${newtest}4.txt
  diff ${LOCAL_TMP_DIR}/${newtest}4.txt ${LOCAL_TEST_DIR}/unit_test_outputs/${newtest}4.txt || die "comparing ${newtest}4.txt" $?

popd

exit 0
