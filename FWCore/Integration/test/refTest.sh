#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/RefTest_cfg.py || die 'Failed in RefTest_cfg.py' $?

# Pass in name and status



