#!/bin/bash
function die { echo $1: status $2; exit $2; }

(cmsRun ${LOCAL_TEST_DIR}/test_prompt_PPSAlCaReco_output.py) || die 'failed running test_prompt_PPSAlCaReco_output.py' $?
