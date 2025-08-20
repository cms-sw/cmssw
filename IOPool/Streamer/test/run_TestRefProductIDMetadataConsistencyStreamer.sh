#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }
function runSuccess {
    echo "cmsRun $@"
    cmsRun $@ || die "cmsRun $*" $?
    echo
}
function runFailure {
    echo "cmsRun $@ (expected to fail)"
    cmsRun $@ && die "cmsRun $*" 1
    echo
}

# Produce two files with different ProductID metadata
runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyStreamer_cfg.py
runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyStreamer_cfg.py --enableOther

# Processing the two files together works
runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyStreamerTest_cfg.py --input refconsistency_1.dat --input refconsistency_10.dat

# Concatenating the two files by keeping the Init message of only first file ...
echo "Concatenating streamer files"
CatStreamerFiles refconsistency_cat.dat refconsistency_1.dat refconsistency_10.dat
echo

# ... fails
runSuccess ${SCRAM_TEST_PATH}/testRefProductIDMetadataConsistencyStreamerTest_cfg.py --input refconsistency_cat.dat
