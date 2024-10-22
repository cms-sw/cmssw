#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${SCRAM_TEST_PATH}/testMerge_cfg.py
F2=${SCRAM_TEST_PATH}/testNoMerge_cfg.py


(cmsRun $F1) > test_merge.log || die "Failure using $F1" $?

diff ${SCRAM_TEST_PATH}/testMerge.log test_merge.log || die "comparing test_merge.log" $?


(cmsRun $F2) > test_no_merge.log || die "Failure using $F2" $?
diff ${SCRAM_TEST_PATH}/testNoMerge.log test_no_merge.log || die "comparing test_no_merge.log" $?
