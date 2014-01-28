#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/inputRawSourceTest_cfg.py || die 'Failed in inputRawSourceTest_cfg.py' $?

# Pass in name and status



