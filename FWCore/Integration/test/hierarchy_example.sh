#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/hierarchy_example_cfg.py || die 'Failed in hierarchy_example_cfg.py' $?

