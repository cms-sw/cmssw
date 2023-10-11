#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${SCRAM_TEST_PATH}/service_example_cfg.py || die 'Failed in service_example_cfg.py' $?
