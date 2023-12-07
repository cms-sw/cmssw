#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

function runTests {

    # local variables
    local testType="$1"
    local inputFiles="$2"
    local sequenceType="$3"
    local isRECO="$4"
    local globalTag="$5"

    echo -e "TESTING step1 ($testType) ...\n\n"

    # optional for the cmsRun sequence
    local sequenceArg=""
    [ -n "$sequenceType" ] && sequenceArg="sequenceType=$sequenceType"
    local globalTagArg=""
    [ -n "$globalTag" ] && globalTagArg="globalTag=$globalTag"

    cmsRun ${SCRAM_TEST_PATH}/Tracker_DataMCValidation_cfg.py maxEvents=100 inputFiles="$inputFiles" $sequenceArg isRECO="$isRECO" $globalTagArg || die "Failure running Tracker_DataMCValidation_cfg.py sequenceType=$sequenceType" $?

    mv step1_DQM_1.root "step1_DQM_1_${testType}.root"

    echo -e "TESTING step2 ($testType)...\n\n"
    cmsRun ${SCRAM_TEST_PATH}/Tracker_DataMCValidation_Harvest_cfg.py inputFiles="file:step1_DQM_1_${testType}.root" || die "Failure running Tracker_DataMCValidation_Harvest_cfg.py" $?

    mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root "step2_DQM_${testType}.root"

    echo -e "================== Done with testing $testType ==================\n\n"
}

#######################################################
# RECO checks
#######################################################
echo "TESTING Tracking DATA/MC comparison codes on RECO ..."

runTests "electrons" "/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/c02ca5ba-f454-4cd3-b114-b55e0309f9db.root" "" "True"
runTests "muons" "/store/relval/CMSSW_13_3_0_pre2/RelValZMM_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/4096bfe7-bc10-4f7f-81ab-4f4adb59e838.root" "muons" "True"
runTests "ttbar" "/store/relval/CMSSW_13_3_0_pre2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/fc1ccb5e-b038-45f2-a06b-e26a6a01681e.root" "ttbar" "True"
runTests "minbias" "/store/relval/CMSSW_13_3_0_pre2/RelValNuGun/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/bc506605-d659-468e-b75a-5d3de82e579f.root" "minbias" "True"
runTests "V0s" "/store/relval/CMSSW_13_3_0_pre2/RelValNuGun/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/bc506605-d659-468e-b75a-5d3de82e579f.root" "V0s" "True"

#######################################################
# AOD checks
#######################################################
echo "TESTING Tracking DATA/MC comparison codes on AOD..."

runTests "electrons (AOD)" "/store/relval/CMSSW_13_0_12/RelValZEE_PU_13p6/AODSIM/PU_130X_mcRun3_2023_realistic_postBPix_v2_RV201-v1/2580000/0d49e310-e06f-4c26-a637-1116b02ef1ce.root" "" "False" "130X_mcRun3_2023_realistic_postBPix_v2"
runTests "muons (AOD)" "/store/relval/CMSSW_13_0_12/RelValZMM_PU_13p6/AODSIM/PU_130X_mcRun3_2023_realistic_postBPix_v2_RV201-v1/2580000/d2a2506c-8954-464b-beda-48242472406d.root" "muons" "False" "130X_mcRun3_2023_realistic_postBPix_v2"
runTests "ttbar (AOD)" "/store/relval/CMSSW_13_0_12/RelValTTbar_SemiLeptonic_PU_13p6/AODSIM/PU_130X_mcRun3_2023_realistic_postBPix_v2_RV201-v1/2580000/08c015c3-c9bd-4017-b21d-264dbaa06445.root" "ttbar" "False" "130X_mcRun3_2023_realistic_postBPix_v2"
runTests "minbias (AOD)" "/store/relval/CMSSW_13_0_12/RelValSingleNuGun_E10_PU/AODSIM/PU_130X_mcRun3_2023_realistic_postBPix_v2_RV201-v1/2580000/37ee5a61-8896-4eb3-8e6c-20ed0ad5b2dc.root" "minbias" "False" "130X_mcRun3_2023_realistic_postBPix_v2"
runTests "V0s (AOD)" "/store/relval/CMSSW_13_0_12/RelValSingleNuGun_E10_PU/AODSIM/PU_130X_mcRun3_2023_realistic_postBPix_v2_RV201-v1/2580000/37ee5a61-8896-4eb3-8e6c-20ed0ad5b2dc.root" "V0s" "False" "130X_mcRun3_2023_realistic_postBPix_v2"
