#!/bin/bash

function die { echo $1: status $2 ;  exit $2; }
function runSuccess {
    echo "cmsRun $@"
    cmsRun $@ || die "cmsRun $*" $?
    echo
}


# Check that changing hardware resources does not lead to new lumi or run
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --accelerators test-one --firstEvent 1 --output test-one.root
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --accelerators test-two --firstEvent 101 --output test-two.root

edmProvDump --hardware test-one.root | grep -q "PROD.*test-one" || die "Did not find test-one from test-one.root provenance" $?
edmProvDump --hardware test-two.root | grep -q "PROD.*test-two" || die "Did not find test-two from test-two.root provenance" $?

runSuccess ${SCRAM_TEST_PATH}/test_merge_two_files.py test-one.root test-two.root

runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged_files.root

edmProvDump --hardware merged_files.root | grep -q "PROD.*test-one" || die "Did not find test-one from merged_files.root provenance" $?
edmProvDump --hardware merged_files.root | grep -q "PROD.*test-two" || die "Did not find test-two from merged_files.root provenance" $?


exit 0
