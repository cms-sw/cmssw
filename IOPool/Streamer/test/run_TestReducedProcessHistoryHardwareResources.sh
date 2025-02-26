#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }
function runSuccess {
    echo "cmsRun $@"
    cmsRun $@ || die "cmsRun $*" $?
    echo
}

# Check that changing hardware resources does not lead to new lumi or run
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --accelerators test-one --firstEvent 1 --output test-one.dat
runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistoryCreate_cfg.py --accelerators test-two --firstEvent 101 --output test-two.dat

CatStreamerFiles merged.dat test-one.dat test-two.dat

runSuccess ${SCRAM_TEST_PATH}/testReducedProcessHistory_cfg.py --input merged.dat --output merged.root

edmProvDump merged.root | grep -q "PROD.*test-one" || die "Did not find test-one from merged.root provenance" $?
edmProvDump merged.root | grep -q "PROD.*test-two" || die "Did not find test-two from merged.root provenance" $?

exit 0
