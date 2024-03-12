#!/bin/bash
function die { echo $1: status $2; exit $2; }

cmsRun ${SCRAM_TEST_PATH}/test_prompt_PPSAlCaReco_output.py inputFiles=file:outputALCAPPS_RECO_prompt_test0.root runNo=355207 || die 'failed running test_prompt_PPSAlCaReco_output.py' $?
cmsRun ${SCRAM_TEST_PATH}/test_prompt_PPSAlCaReco_output.py inputFiles=file:outputALCAPPS_RECO_prompt_test1.root runNo=367104 || die 'failed running test_prompt_PPSAlCaReco_output.py' $?
