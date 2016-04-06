#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python $1 || die "Failure for configuration: $1" $?; }


runTest "${LOCAL_TEST_DIR}/RunRepack.py --select-events HLT:path1,HLT:path2 --lfn /store/whatever"

declare -a arr=("cosmics" "pp" "cosmicsRun2" "cosmicsEra_Run2_25ns" "cosmicsEra_Run2_2016" "ppRun2" "ppRun2B0T" "ppRun2at50ns" "ppEra_Run2_50ns" "ppEra_Run2_25ns" "ppEra_Run2_2016" "pplowpuEra_Run2_2016" "ppEra_Run2_2016_trackingLowPU")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario $scenario --global-tag GLOBALTAG --lfn /store/whatever --fevt --dqmio  --alcareco TkAlMinBias+SiStripCalMinBias "
     runTest "${LOCAL_TEST_DIR}/RunVisualizationProcessing.py --scenario $scenario --lfn /store/whatever --global-tag GLOBALTAG --fevt"
     runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario $scenario --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --workflows=BeamSpotByRun,BeamSpotByLumi,SiStripQuality"
done


declare -a arr=("cosmics" "pp" "cosmicsRun2" "cosmicsEra_Run2_25ns" "cosmicsEra_Run2_2016" "ppRun2" "AlCaLumiPixels" "AlCaTestEnable" "hcalnzs" "hcalnzsRun2" "hcalnzsEra_Run2_25ns" "hcalnzsEra_Run2_2016" "pplowpuEra_Run2_2016" "ppEra_Run2_2016_trackingLowPU" "ppRun2B0T" "ppRun2at50ns" "ppEra_Run2_50ns" "ppEra_Run2_25ns" "ppEra_Run2_2016")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias"
done

declare -a arr=("HeavyIonsRun2" "HeavyIonsEra_Run2_HI")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario $scenario --global-tag GLOBALTAG --lfn /store/whatever --fevt --dqmio  --alcareco TkAlMinBiasHI+SiStripCalMinBias "
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBiasHI+SiStripCalMinBias"
done


declare -a arr=("cosmics" "pp" "cosmicsRun2" "cosmicsEra_Run2_25ns" "cosmicsEra_Run2_2016" "ppRun2" "HeavyIons" "HeavyIonsRun2" "HeavyIonsEra_Run2_HI" "AlCaLumiPixels" "ppRun2B0T" "ppRun2at50ns" "ppEra_Run2_50ns" "ppEra_Run2_25ns" "ppEra_Run2_2016" "pplowpuEra_Run2_2016" "ppEra_Run2_2016_trackingLowPU")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario $scenario --lfn=/store/whatever --global-tag GLOBALTAG --skims SiStripCalZeroBias,SiStripCalMinBias,PromptCalibProd"
     runTest "${LOCAL_TEST_DIR}/RunDQMHarvesting.py --scenario $scenario --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG"
done

declare -a arr=("ppRun2" "ppRun2B0T" "ppRun2at50ns" "ppEra_Run2_50ns" "ppEra_Run2_25ns" "ppEra_Run2_2016" "pplowpuEra_Run2_2016" "ppEra_Run2_2016_trackingLowPU")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --miniaod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias"
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias --PhysicsSkim=@SingleMuon"
done







