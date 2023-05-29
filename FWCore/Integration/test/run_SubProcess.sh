#!/bin/bash

test=testSubProcess

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

  rm -f testSubProcess.grep.txt
  rm -f ${test}.log

  echo cmsRun testSubProcess_cfg.py
  cmsRun -p ${LOCAL_TEST_DIR}/${test}_cfg.py >& ${test}.log 2>&1 || die "cmsRun ${test}_cfg.py" $?
  grep Doodad ${test}.log > testSubProcess.grep.txt
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSubProcess.grep.txt testSubProcess.grep.txt || die "comparing testSubProcess.grep.txt" $?
  grep "^++" ${test}.log | grep -v "Disabling gnu" > testSubProcess.grep2.txt
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSubProcess.grep2.txt testSubProcess.grep2.txt || die "comparing testSubProcess.grep2.txt" $?

  echo cmsRun readSubProcessOutput_cfg.py
  cmsRun -p ${LOCAL_TEST_DIR}/readSubProcessOutput_cfg.py || die "cmsRun readSubProcessOutput_cfg.py" $?


  echo cmsRun testSubProcessEventSetup_cfg.py
  cmsRun -p ${LOCAL_TEST_DIR}/testSubProcessEventSetup_cfg.py > testSubProcessEventSetup.log 2>&1 || die "cmsRun testSubProcessEventSetup_cfg.py" $?
  grep "Sharing" testSubProcessEventSetup.log > testSubProcessEventSetup.grep.txt
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSubProcessEventSetup.grep.txt  testSubProcessEventSetup.grep.txt || die "comparing testSubProcessEventSetup.grep.txt" $?

  echo cmsRun testSubProcessEventSetup1_cfg.py
  cmsRun -p ${LOCAL_TEST_DIR}/testSubProcessEventSetup1_cfg.py > testSubProcessEventSetup1.log 2>&1 || die "cmsRun testSubProcessEventSetup1_cfg.py" $?
  grep "ESTestAnalyzerB: p" testSubProcessEventSetup1.log > testSubProcessEventSetup1.grep.txt
  grep "ESTestAnalyzerK: p" testSubProcessEventSetup1.log >> testSubProcessEventSetup1.grep.txt
  grep "Sharing" testSubProcessEventSetup1.log >> testSubProcessEventSetup1.grep.txt
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSubProcessEventSetup1.grep.txt  testSubProcessEventSetup1.grep.txt || die "comparing testSubProcessEventSetup1.grep.txt" $?

  echo cmsRun testSubProcessUnscheduled_cfg.py
  cmsRun -p ${LOCAL_TEST_DIR}/testSubProcessUnscheduled_cfg.py || die "cmsRun testSubProcessUnscheduled_cfg.py" $?

  echo cmsRun testSubProcessUnscheduledRead_cfg.py
  cmsRun -p ${LOCAL_TEST_DIR}/testSubProcessUnscheduledRead_cfg.py || die "cmsRun testSubProcessUnscheduledRead_cfg.py" $?

exit 0
