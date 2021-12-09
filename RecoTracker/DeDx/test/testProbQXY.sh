#!/bin/sh

function die { echo $1: status $2 ; exit $2; }

echo "TESTING Getting probQ and probXY from Analysis Data-Tier codes ..."
echo -e "For now not doing anything, when there will be AOD and MiniAOD with this variable, this can be uncommented"
# test AOD
# Will come soon
#echo "TESTING refitting from AOD ...\n\n"
#cmsRun ${LOCAL_TEST_DIR}/ProbQXYFromMiniAOD.py maxEvents=100 inputFiles=/store/mc/RunIISummer20UL16RECO/DYJetsToMuMu_M-50_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/AODSIM/106X_mcRun2_asymptotic_v13-v2/00000/1D4FBF4B-7B8D-A04F-A108-B1BEB60558FA.root || die "Failure refitting from AOD" $?

# test MINIAOD
echo -e "TESTING  Getting probQ and probXY from MINIAOD ...\n\n"
cmsDriver.py ReRECO --conditions auto:run2_data --customise Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018 --datatier MINIAOD --eventcontent MINIAOD --number 10 --python_filename 4HLT2MiniAOD.py --scenario pp --step RAW2DIGI,L1Reco,RECO,PAT --data  --filein /store/data/Run2018C/SingleMuon/RAW/v1/000/319/337/00000/004ED286-5582-E811-8895-FA163E126126.root --fileout file:MiniAOD.root --era Run2_2018  --runUnscheduled 

cmsRun ${LOCAL_TEST_DIR}/ProbQXYFromMiniAOD_cfg.py maxEvents=10 inputFiles=file:MiniAOD.root || die "Failure getting probQ and probXY from MINIAOD" $?
rm MiniAOD.root

# test MINIAOD signal MC
echo -e "TESTING  Getting probQ and probXY from signal MC MINIAOD ...\n\n"
cmsDriver.py ReRECOonMC --conditions 106X_upgrade2018_realistic_v11_L1v1 --datatier MINIAOD --era Run2_2018,pf_badHcalMitigation --eventcontent MINIAOD --number 10 --python_filename 4HLT2MiniAODonMC.py --scenario pp --step RAW2DIGI,L1Reco,RECO,EI,PAT --runUnscheduled --mc  --filein  /store/user/tvami/HSCP/HSCPgluino_M_1800/crab_PrivateHSCP_2018_Gluino_Mass1800_HLT_NoPU_v1/210612_042006/0000/HSCP_Gluino_Mass1800_HLT_1.root  --fileout file:MiniAODonMC.root

cmsRun ${LOCAL_TEST_DIR}/ProbQXYFromMiniAOD_cfg.py maxEvents=10 inputFiles=file:MiniAODonMC.root || die "Failure getting probQ and probXY from Signal MC MINIAOD" $?
rm MiniAODonMC.root

