#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python $1 || die "Failure for configuration: $1" $?; }


runTest "${LOCAL_TEST_DIR}/RunRepack.py --select-events HLT:path1,HLT:path2 --lfn /store/whatever"

declare -a arr=("cosmics" "pp" "cosmicsEra_Run2_25ns" "cosmicsEra_Run2_2016" "ppEra_Run2_50ns" "ppEra_Run2_25ns" "ppEra_Run2_2016" "ppEra_Run2_2016_trackingLowPU" "ppEra_Run2_2016_pA" "cosmicsEra_Run2_2017" "ppEra_Run2_2017" "ppEra_Run2_2017_trackingLowPU" "ppEra_Run2_2017_pp_on_XeXe" "ppEra_Run2_2017_ppRef")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario $scenario --global-tag GLOBALTAG --lfn /store/whatever --fevt --dqmio  --alcareco TkAlMinBias+SiStripCalMinBias "
     runTest "${LOCAL_TEST_DIR}/RunVisualizationProcessing.py --scenario $scenario --lfn /store/whatever --global-tag GLOBALTAG --fevt"
     runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario $scenario --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --workflows=BeamSpotByRun,BeamSpotByLumi,SiStripQuality"
done


declare -a arr=("cosmics" "pp" "cosmicsEra_Run2_25ns" "cosmicsEra_Run2_2016" "AlCaLumiPixels" "AlCaTestEnable" "hcalnzs" "hcalnzsEra_Run2_25ns" "hcalnzsEra_Run2_2016" "ppEra_Run2_2016_trackingLowPU" "ppEra_Run2_2016_pA" "ppEra_Run2_50ns" "ppEra_Run2_25ns" "ppEra_Run2_2016" "cosmicsEra_Run2_2017" "hcalnzsEra_Run2_2017" "ppEra_Run2_2017_trackingLowPU" "ppEra_Run2_2017_pp_on_XeXe" "ppEra_Run2_2017_ppRef" "ppEra_Run2_2017")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias"
done

declare -a arr=("HeavyIonsEra_Run2_HI")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario $scenario --global-tag GLOBALTAG --lfn /store/whatever --fevt --dqmio  --alcareco TkAlMinBiasHI+SiStripCalMinBias "
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBiasHI+SiStripCalMinBias"
done


declare -a arr=("cosmics" "pp" "cosmicsEra_Run2_25ns" "cosmicsEra_Run2_2016" "HeavyIons" "HeavyIonsEra_Run2_HI" "AlCaLumiPixels" "ppEra_Run2_50ns" "ppEra_Run2_25ns" "ppEra_Run2_2016" "ppEra_Run2_2016_trackingLowPU" "ppEra_Run2_2016_pA" "cosmicsEra_Run2_2017" "ppEra_Run2_2017_trackingLowPU" "ppEra_Run2_2017_pp_on_XeXe" "ppEra_Run2_2017_ppRef" "ppEra_Run2_2017")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario $scenario --lfn=/store/whatever --global-tag GLOBALTAG --skims SiStripCalZeroBias,SiStripCalMinBias,PromptCalibProd"
     runTest "${LOCAL_TEST_DIR}/RunDQMHarvesting.py --scenario $scenario --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG"
done

declare -a arr=("ppEra_Run2_50ns" "ppEra_Run2_25ns" "ppEra_Run2_2016" "ppEra_Run2_2016_trackingLowPU" "ppEra_Run2_2016_pA" "ppEra_Run2_2017" "ppEra_Run2_2017_trackingLowPU" "ppEra_Run2_2017_pp_on_XeXe" "ppEra_Run2_2017_ppRef")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --miniaod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias"
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias --PhysicsSkim=@SingleMuon"
done

runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario AlCaTestEnable --global-tag GLOBALTAG --lfn /store/whatever --alcareco PromptCalibProdEcalPedestals "
runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario AlCaTestEnable --lfn=/store/whatever --global-tag GLOBALTAG --skims PromptCalibProdEcalPedestals"
runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario AlCaTestEnable --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --workflows=EcalPedestals"

runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario ppEra_Run2_2017_trackingOnly --global-tag GLOBALTAG  --lfn /store/whatever  --alcarecos=TkAlMinBias"
runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario ppEra_Run2_2017_trackingOnly --lfn /store/whatever --global-tag GLOBALTAG --skims TkAlMinBias"
runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario ppEra_Run2_2017_trackingOnly --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG"
