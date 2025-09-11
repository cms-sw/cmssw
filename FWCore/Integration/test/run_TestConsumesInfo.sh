#!/bin/bash

test=testGetBy

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

  echo "testConsumesInfo"
  cmsRun ${LOCAL_TEST_DIR}/testConsumesInfo_cfg.py > testConsumesInfo.log 2>/dev/null || die "cmsRun testConsumesInfo_cfg.py" $?
  grep -v '++\|LegacyModules\|time' testConsumesInfo.log > testConsumesInfo_1.log
  rm testConsumesInfo.log
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/testConsumesInfo_1.log testConsumesInfo_1.log || die "comparing testConsumesInfo_1.log" $?

exit 0
