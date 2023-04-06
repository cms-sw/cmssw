#!/bin/sh


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }
cmsRun ${SCRAM_TEST_PATH}/write_cfg.py || die 'failed to write conditions' $?
cmsRun ${SCRAM_TEST_PATH}/read_cfg.py || die 'failed to read conditions' $?

cmsRun ${SCRAM_TEST_PATH}/write2_cfg.py || die 'failed to write conditions by discovering what is in a record' $?
cmsRun ${SCRAM_TEST_PATH}/read_cfg.py || die 'failed to read conditions which used auto discovery' $?

