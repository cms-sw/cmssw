#!/bin/bash

NAME=$1
VALUE=$2

function die { echo $1: status $2 ; echo === Log file === ; cat ${3:-/dev/null} ; echo === End log file === ; exit $2; }

cmsRun ${SCRAM_TEST_PATH}/test_wrongOptionsType_cfg.py --name=${NAME} --value="$VALUE" > ${NAME}.log 2>&1 && die "cmsRun for ${NAME} succeeded, should have failed" 1 ${NAME}.log
grep -E "(The type in the configuration is incorrect)|(ValueError type of .* is expected to be .* but declared as)" ${NAME}.log > /dev/null || die "cmsRun for ${NAME} failed for other reason than incorrect configuration type" $? ${NAME}.log
