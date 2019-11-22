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

  #It is intentional that this cmsRun process throws an exception
  echo "testDuplicateProcess"
  cmsRun ${LOCAL_TEST_DIR}/testDuplicateProcess_cfg.py &> testDuplicateProcess.log && die 'Failed to get exception running testDuplicateProcess_cfg.py' 1
  grep -q "Duplicate Process" testDuplicateProcess.log || die 'Failed to print out exception message for duplicate process name' $?

  echo "testGetBy1Mod"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}1Mod_cfg.py > testGetBy1Mod.log 2>/dev/null || die "cmsRun ${test}1Mod_cfg.py" $?

  echo "testGetByMerge"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}Merge_cfg.py > testGetByMerge.log 2>/dev/null || die "cmsRun ${test}Merge_cfg.py" $?

  echo "testGetByPlaceholder"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}Placeholder_cfg.py || die "cmsRun ${test}Placeholder_cfg.py" $?

  echo "testProducesCollector"
  cmsRun -p ${LOCAL_TEST_DIR}/testProducesCollector_cfg.py || die "cmsRun testProducesCollector_cfg.py" $?

popd

exit 0
