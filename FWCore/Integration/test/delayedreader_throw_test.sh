#!/bin/bash
set -x

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun ${SCRAM_TEST_PATH}/delayedreader_throw_cfg.py && die "cmsRun delayedreader_throw_cfg.py did not fail" 1

exit 0
