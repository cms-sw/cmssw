#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun -p ${SCRAM_TEST_PATH}/testLumiProd_cfg.py || die "cmsRun testLumiProd_cfg.py" $?
