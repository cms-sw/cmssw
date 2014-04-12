#!/bin/bash

test=testGetBy

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  cmsRun -p ${LOCAL_TEST_DIR}/${test}1_cfg.py > testGetBy1.log || die "cmsRun ${test}1_cfg.py" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testGetBy1.log testGetBy1.log || die "comparing testGetBy1.log" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}2_cfg.py > testGetBy2.log || die "cmsRun ${test}2_cfg.py" $?
  grep -v "Initiating request to open file" testGetBy2.log | grep -v "Successfully opened file" | grep -v "Closed file" > testGetBy2_1.log
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testGetBy2.log testGetBy2_1.log || die "comparing testGetBy2.log" $?

  cmsRun -p ${LOCAL_TEST_DIR}/${test}3_cfg.py || die "cmsRun ${test}3_cfg.py" $?

popd

exit 0
