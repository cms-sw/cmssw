#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

LOCAL_TEST_DIR=$SCRAM_TEST_PATH

  cmsRun ${LOCAL_TEST_DIR}/testEdmConfigDump_cfg.py > testEdmConfigDump.log || die "cmsRun testEdmConfigDump_cfg.py" $?
  edmProvDump testEdmProvDump.root > provdump.log
  python3 ${LOCAL_TEST_DIR}/removeChangingParts.py provdump.log > provdump1.log
  diff ${LOCAL_TEST_DIR}/unit_test_outputs/provdump.log provdump1.log  || die "comparing provdump.log" $?

exit 0
