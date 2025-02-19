#!/bin/bash

test=testSubProcess

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  rm -f testSubProcess.grep.txt
  rm -f ${test}.log

  cmsRun -p ${LOCAL_TEST_DIR}/${test}_cfg.py > ${test}.log 2>&1 || die "cmsRun ${test}_cfg.py" $?
  grep Doodad ${test}.log > testSubProcess.grep.txt
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testSubProcess.grep.txt testSubProcess.grep.txt || die "comparing testSubProcess.grep.txt" $?

  cmsRun -p ${LOCAL_TEST_DIR}/readSubProcessOutput_cfg.py || die "cmsRun readSubProcessOutput_cfg.py" $?

popd

exit 0
