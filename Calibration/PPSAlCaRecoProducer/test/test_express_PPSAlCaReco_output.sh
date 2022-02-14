#!/bin/bash
function die { echo $1: status $2; exit $2; }

(cmsRun ${LOCAL_TEST_DIR}/test_express_PPSAlCaReco_output.py) || die 'failed running test_express_PPSAlCaReco_output.py' $?
