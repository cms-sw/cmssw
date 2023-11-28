#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

export STD_OUT=./cout.txt
export STD_ERR=./cerr.txt

#(totalview cmsRun -a ${SCRAM_TEST_PATH}/messageLogger_cfg.py > ${STD_OUT}  2> ${STD_ERR} ) || die 'Failure using messageLogger_cfg.py' $?
(cmsRun ${SCRAM_TEST_PATH}/messageLogger_cfg.py > ${STD_OUT}  2> ${STD_ERR} ) || die 'Failure using messageLogger_cfg.py' $?
