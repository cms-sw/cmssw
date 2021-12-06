#!/bin/bash

check_for_success() {
    "${@}" && echo -e "\n ---> Passed test of '${@}'\n\n" || exit 1
}

function die { echo $1: status $2; exit $2; }

########################################
# Test help function
########################################
check_for_success run_AlCaRecoTriggerBitsUpdateWorkflow.py --help

########################################
# Test update AlCaRecoTriggerBits
########################################
run_AlCaRecoTriggerBitsUpdateWorkflow.py -f frontier://PromptProd/CMS_CONDITIONS -i AlCaRecoHLTpaths8e29_1e31_v24_offline -d AlCaRecoHLTpaths_TEST || die 'failed running run_AlCaRecoTriggerBitsUpdateWorkflow.py' $?

########################################
# Test read AlCaRecoTriggerBits
########################################
cmsRun $CMSSW_BASE/src/CondTools/HLT/test/AlCaRecoTriggerBitsRcdRead_TEMPL_cfg.py inputDB=sqlite_file:AlCaRecoHLTpaths_TEST.db inputTag=AlCaRecoHLTpaths_TEST || die 'failed running AlCaRecoTriggerBitsRcdRead_TEMPL_cfg.py' $?
