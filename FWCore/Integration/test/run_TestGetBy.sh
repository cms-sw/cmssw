#!/bin/bash

test=testGetBy

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  echo "testGetBy1"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}1_cfg.py > testGetBy1.log 2>/dev/null || die "cmsRun ${test}1_cfg.py" $?
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testGetBy1.log testGetBy1.log || die "comparing testGetBy1.log" $?

  echo "testGetBy2"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}2_cfg.py > testGetBy2.log 2>/dev/null || die "cmsRun ${test}2_cfg.py" $?
  grep -v "Initiating request to open file" testGetBy2.log | grep -v "Successfully opened file" | grep -v "Closed file" > testGetBy2_1.log
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testGetBy2.log testGetBy2_1.log || die "comparing testGetBy2.log" $?

  echo "testGetBy3"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}3_cfg.py || die "cmsRun ${test}3_cfg.py" $?

  echo "testConsumesInfo"
  cmsRun -p ${LOCAL_TEST_DIR}/testConsumesInfo_cfg.py > testConsumesInfo.log 2>/dev/null || die "cmsRun testConsumesInfo_cfg.py" $?
  grep -v "++" testConsumesInfo.log > testConsumesInfo_1.log
  rm testConsumesInfo.log
  rm testConsumesInfo.root
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testConsumesInfo_1.log testConsumesInfo_1.log || die "comparing testConsumesInfo_1.log" $?

popd

exit 0
