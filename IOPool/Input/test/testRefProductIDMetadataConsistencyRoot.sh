#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }
function runSuccess {
    echo "cmsRun $@"
    cmsRun $@ || die "cmsRun $*" $?
    echo
}

runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyRoot_cfg.py
runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyRoot_cfg.py --enableOther
runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyRootMerge_cfg.py
runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyRootTest_cfg.py
