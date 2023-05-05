#!/bin/bash

test=testGetBy

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

  echo "testGetBy1"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}1_cfg.py > testGetBy1.log 2>/dev/null || die "cmsRun ${test}1_cfg.py" $?
  grep -v "LegacyModules" testGetBy1.log > testGetBy1_1.log
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testGetBy1.log testGetBy1_1.log || die "comparing testGetBy1.log" $?

  echo "testGetBy2"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}2_cfg.py > testGetBy2.log 2>/dev/null || die "cmsRun ${test}2_cfg.py" $?
  grep -v 'Initiating request to open file\|Successfully opened file\|Closed file\|LegacyModules' testGetBy2.log > testGetBy2_1.log
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testGetBy2.log testGetBy2_1.log || die "comparing testGetBy2.log" $?

  echo "testGetBy3"
  cmsRun -p ${LOCAL_TEST_DIR}/${test}3_cfg.py || die "cmsRun ${test}3_cfg.py" $?

  echo "testConsumesInfo"
  cmsRun -p ${LOCAL_TEST_DIR}/testConsumesInfo_cfg.py > testConsumesInfo.log 2>/dev/null || die "cmsRun testConsumesInfo_cfg.py" $?
  grep -v '++\|LegacyModules' testConsumesInfo.log > testConsumesInfo_1.log
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

  echo "testGetByRunsMode_cfg.py"
  cmsRun -p ${LOCAL_TEST_DIR}/testGetByRunsMode_cfg.py || die "cmsRun testGetByRunsMode_cfg.py" $?

  echo "testGetByRunsLumisMode_cfg.py"
  cmsRun -p ${LOCAL_TEST_DIR}/testGetByRunsLumisMode_cfg.py || die "cmsRun testGetByRunsLumisMode_cfg.py" $?

  echo "testGetByWithEmptyRun_cfg.py"
  cmsRun -p ${LOCAL_TEST_DIR}/testGetByWithEmptyRun_cfg.py || die "cmsRun testGetByWithEmptyRun_cfg.py" $?

  echo "testGetterOfProductsWithOutputModule_cfg.py"
  cmsRun -p ${LOCAL_TEST_DIR}/testGetterOfProductsWithOutputModule_cfg.py || die "cmsRun testGetterOfProductsWithOutputModule_cfg.py" $?

exit 0
