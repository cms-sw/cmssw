#!/bin/sh


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }
cmsRun ${LOCAL_TEST_DIR}/write_cfg.py || die 'failed to write conditions' $?
cmsRun ${LOCAL_TEST_DIR}/read_cfg.py || die 'failed to read conditions' $?

cmsRun ${LOCAL_TEST_DIR}/write2_cfg.py || die 'failed to write conditions by discovering what is in a record' $?
cmsRun ${LOCAL_TEST_DIR}/read_cfg.py || die 'failed to read conditions which used auto discovery' $?

