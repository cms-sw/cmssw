#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Tracking DATA/MC comparison codes ..."

echo -e "TESTING step1 (electrons) ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/Tracker_DataMCValidation_cfg.py maxEvents=100 inputFiles=/store/relval/CMSSW_13_3_0_pre2/RelValZEE_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/c02ca5ba-f454-4cd3-b114-b55e0309f9db.root || die "Failure running Tracker_DataMCValidation_cfg.py" $?

mv step1_DQM_1.root step1_DQM_1_electrons.root

echo -e "TESTING step2 (electrons)...\n\n"
cmsRun ${SCRAM_TEST_PATH}/Tracker_DataMCValidation_Harvest_cfg.py inputFiles=file:step1_DQM_1_electrons.root || die "Failure running Tracker_DataMCValidation_Harvest_cfg.py" $?

echo -e "================== Done with testing electrons ==================\n\n"

echo -e "TESTING step1 (muons) ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/Tracker_DataMCValidation_cfg.py maxEvents=100 inputFiles=/store/relval/CMSSW_13_3_0_pre2/RelValZMM_14/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/4096bfe7-bc10-4f7f-81ab-4f4adb59e838.root sequenceType=muons || die "Failure running Tracker_DataMCValidation_cfg.py sequenceType=muons" $?

mv step1_DQM_1.root step1_DQM_1_muons.root

echo -e "TESTING step2 (muons)...\n\n"
cmsRun ${SCRAM_TEST_PATH}/Tracker_DataMCValidation_Harvest_cfg.py inputFiles=file:step1_DQM_1_muons.root || die "Failure running Tracker_DataMCValidation_Harvest_cfg.py" $?

echo -e "================== Done with testing muons ==================...\n\n"

echo -e "TESTING step1 (ttbar) ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/Tracker_DataMCValidation_cfg.py maxEvents=100 inputFiles=/store/relval/CMSSW_13_3_0_pre2/RelValTTbar_14TeV/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/fc1ccb5e-b038-45f2-a06b-e26a6a01681e.root sequenceType=ttbar || die "Failure running Tracker_DataMCValidation_cfg.py sequenceType=ttbar" $?

mv step1_DQM_1.root step1_DQM_1_ttbar.root

echo -e "TESTING step2 (ttbar)...\n\n"
cmsRun ${SCRAM_TEST_PATH}/Tracker_DataMCValidation_Harvest_cfg.py inputFiles=file:step1_DQM_1_ttbar.root || die "Failure running Tracker_DataMCValidation_Harvest_cfg.py" $?

echo -e "================== Done with testing ttbar ==================...\n\n"

echo "TESTING step1 (minbias) ...\n\n"
cmsRun ${SCRAM_TEST_PATH}/Tracker_DataMCValidation_cfg.py maxEvents=100 inputFiles=/store/relval/CMSSW_13_3_0_pre2/RelValNuGun/GEN-SIM-RECO/PU_132X_mcRun3_2023_realistic_v2_RV213-v1/2580000/bc506605-d659-468e-b75a-5d3de82e579f.root sequenceType=minbias || die "Failure running Tracker_DataMCValidation_cfg.py sequenceType=ttbar" $?

mv step1_DQM_1.root step1_DQM_1_minbias.root

echo "TESTING step2 (minbias)...\n\n"
cmsRun ${SCRAM_TEST_PATH}/Tracker_DataMCValidation_Harvest_cfg.py inputFiles=file:step1_DQM_1_minbias.root || die "Failure running Tracker_DataMCValidation_Harvest_cfg.py" $?

echo -e "================== Done with testing minbias ==================...\n\n"
