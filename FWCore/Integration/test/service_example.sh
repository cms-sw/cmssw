#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${SCRAM_TEST_PATH}/service_example_cfg.py || die 'Failed in service_example_cfg.py' $?
