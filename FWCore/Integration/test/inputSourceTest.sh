#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/inputSourceTest_cfg.py || die 'Failed in inputSourceTest_cfg.py' $?
