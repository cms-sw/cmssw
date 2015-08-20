#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python $1 || die "Failure for configuration: $1" $?; }


runTest "${LOCAL_TEST_DIR}/RunRepack.py --select-events HLT:path1,HLT:path2 --lfn /store/whatever"

declare -a arr=("cosmics" "pp" "cosmicsRun2" "ppRun2" "ppRun2B0T" "HeavyIons" "ppRun2at50ns")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario $scenario --global-tag GLOBALTAG --lfn /store/whatever --fevt --dqmio  --alcareco TkAlMinBias+SiStripCalMinBias "
     runTest "${LOCAL_TEST_DIR}/RunVisualizationProcessing.py --scenario $scenario --lfn /store/whatever --global-tag GLOBALTAG --fevt"
     runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario $scenario --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --workflows=BeamSpotByRun,BeamSpotByLumi,SiStripQuality"
done


declare -a arr=("cosmics" "pp" "cosmicsRun2" "ppRun2" "HeavyIons" "AlCaLumiPixels" "AlCaTestEnable" "hcalnzs" "ppRun2B0T" "ppRun2at50ns")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias"
done


declare -a arr=("cosmics" "pp" "cosmicsRun2" "ppRun2" "HeavyIons" "AlCaLumiPixels" "ppRun2B0T" "ppRun2at50ns")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario $scenario --lfn=/store/whatever --global-tag GLOBALTAG --skims SiStripCalZeroBias,SiStripCalMinBias,PromptCalibProd"
     runTest "${LOCAL_TEST_DIR}/RunDQMHarvesting.py --scenario $scenario --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG"
done

declare -a arr=("ppRun2" "ppRun2B0T" "ppRun2at50ns")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --miniaod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias"
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias --PhysicsSkim=@SingleMuon"
done







