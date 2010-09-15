#!/bin/bash

test=testRunMerge

function die { echo Failure $1: status $2 ; exit $2 ; }

echo LOCAL_TMP_DIR = ${LOCAL_TMP_DIR}

pushd ${LOCAL_TMP_DIR}
  echo ${test}PROD0 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD0_cfg.py || die "cmsRun ${test}PROD0_cfg.py" $?

  echo ${test}PROD1 ------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD1_cfg.py || die "cmsRun ${test}PROD1_cfg.py" $?

  echo ${test}PROD2------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD2_cfg.py || die "cmsRun ${test}PROD2_cfg.py" $?

  echo ${test}PROD2EXTRA------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD2EXTRA_cfg.py || die "cmsRun ${test}PROD2EXTRA_cfg.py" $?

  echo ${test}PROD3------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD3_cfg.py || die "cmsRun ${test}PROD3_cfg.py" $?

  echo ${test}PROD3EXTRA------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD3EXTRA_cfg.py || die "cmsRun ${test}PROD3EXTRA_cfg.py" $?

  echo ${test}PROD4------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD4_cfg.py || die "cmsRun ${test}PROD4_cfg.py" $?

  echo ${test}PROD5------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD5_cfg.py || die "cmsRun ${test}PROD5_cfg.py" $?

  echo ${test}PROD6------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD6_cfg.py || die "cmsRun ${test}PROD6_cfg.py" $?

  echo ${test}PROD7------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PROD7_cfg.py || die "cmsRun ${test}PROD7_cfg.py" $?

  echo ${test}MERGE------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE_cfg.py || die "cmsRun ${test}MERGE_cfg.py" $?

  echo ${test}MERGE1------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE1_cfg.py || die "cmsRun ${test}MERGE1_cfg.py" $?

  echo ${test}MERGE2------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE2_cfg.py || die "cmsRun ${test}MERGE2_cfg.py" $?

  echo ${test}MERGE3------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE3_cfg.py || die "cmsRun ${test}MERGE3_cfg.py" $?

  echo ${test}MERGE3x------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE3x_cfg.py || die "cmsRun ${test}MERGE3x_cfg.py" $?

  echo ${test}MERGE4------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}MERGE4_cfg.py || die "cmsRun ${test}MERGE4_cfg.py" $?

  echo ${test}TEST------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}TEST_cfg.py || die "cmsRun ${test}TEST_cfg.py" $?

  echo ${test}TESTFAIL------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}TESTFAIL_cfg.py 2>/dev/null && die "cmsRun ${test}TESTFAIL_cfg.py" $?

  echo ${test}TEST1------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}TEST1_cfg.py || die "cmsRun ${test}TEST1_cfg.py" $?

  echo ${test}TEST2------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}TEST2_cfg.py || die "cmsRun ${test}TEST2_cfg.py" $?

  echo ${test}TEST3------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}TEST3_cfg.py || die "cmsRun ${test}TEST3_cfg.py" $?

  echo ${test}TEST4------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}TEST4_cfg.py || die "cmsRun ${test}TEST4_cfg.py" $?

  echo ${test}COPY------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}COPY_cfg.py || die "cmsRun ${test}COPY_cfg.py" $?

  echo ${test}COPY1------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}COPY1_cfg.py || die "cmsRun ${test}COPY1_cfg.py" $?

  echo ${test}PickEvents------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PickEvents_cfg.py || die "cmsRun ${test}PickEvents_cfg.py" $?

  echo ${test}PickEventsx------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}PickEventsx_cfg.py || die "cmsRun ${test}PickEventsx_cfg.py" $?

  echo ${test}FastCloning------------------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/${test}FastCloning_cfg.py 2> testFastCloning.txt
  grep "Another exception was caught" testFastCloning.txt || die "cmsRun testRunMergeFastCloning_cfg.py" $?

  echo testLooperEventNavigation-----------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/testLooperEventNavigation_cfg.py < ${LOCAL_TEST_DIR}/testLooperEventNavigation.txt > testLooperEventNavigationOutput.txt || die "cmsRun testLooperEventNavigation_cfg.py " $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testLooperEventNavigationOutput.txt testLooperEventNavigationOutput.txt || die "comparing testLooperEventNavigationOutput.txt" $?

  echo testLooperEventNavigation1-----------------------------------------------------
  cmsRun -p ${LOCAL_TEST_DIR}/testLooperEventNavigation1_cfg.py < ${LOCAL_TEST_DIR}/testLooperEventNavigation.txt > testLooperEventNavigationOutput1.txt || die "cmsRun testLooperEventNavigation1_cfg.py " $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testLooperEventNavigationOutput.txt testLooperEventNavigationOutput1.txt || die "comparing testLooperEventNavigationOutput1.txt" $?

popd

exit 0
