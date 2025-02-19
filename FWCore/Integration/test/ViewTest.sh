#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/ViewTest_cfg.py || die 'Failed in ViewTest_cfg.py' $?
