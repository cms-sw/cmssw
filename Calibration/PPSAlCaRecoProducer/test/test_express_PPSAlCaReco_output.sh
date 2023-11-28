#!/bin/bash
function die { echo $1: status $2; exit $2; }

(cmsRun ${SCRAM_TEST_PATH}/test_express_PPSAlCaReco_output.py) || die 'failed running test_express_PPSAlCaReco_output.py' $?
