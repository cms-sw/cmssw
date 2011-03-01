#!/bin/sh

rm -f ${LOCAL_TEST_DIR}/pfcand_test.root

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/write_cfg.py || die 'Failed to create file' $?
cmsRun ${LOCAL_TEST_DIR}/read_cfg.py || die 'Failed to read file' $?

