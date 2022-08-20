#! /bin/bash

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Alignment/OfflineValidation ..."
cmsRun ${LOCAL_TEST_DIR}/test_all_cfg.py || die "Failure running test_all_cfg.py" $?
