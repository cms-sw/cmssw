#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python $1 || die "Failure for configuration: $1" $?; }


runTest "${LOCAL_TEST_DIR}/RunRepack.py --select-events HLT:path1,HLT:path2 --lfn /store/whatever"

declare -a arr=("cosmicsEra_Run2_2017" "ppEra_Run2_2017" "ppEra_Run2_2017_trackingLowPU" "ppEra_Run2_2017_pp_on_XeXe" "ppEra_Run2_2017_ppRef" "cosmicsEra_Run2_2018" "ppEra_Run2_2018" "ppEra_Run2_2018_highBetaStar" "ppEra_Run2_2018_pp_on_AA" "cosmicsHybridEra_Run2_2018" "cosmicsEra_Run3" "ppEra_Run3")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario $scenario --global-tag GLOBALTAG --lfn /store/whatever --fevt --dqmio  --alcareco TkAlMinBias+SiStripCalMinBias "
     runTest "${LOCAL_TEST_DIR}/RunVisualizationProcessing.py --scenario $scenario --lfn /store/whatever --global-tag GLOBALTAG --fevt"
     runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario $scenario --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --workflows=BeamSpotByRun,BeamSpotByLumi,SiStripQuality"
done

declare -a arr=("HeavyIonsEra_Run2_2018")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario $scenario --global-tag GLOBALTAG --lfn /store/whatever --fevt --dqmio  --alcareco TkAlMinBiasHI+SiStripCalMinBias "
     runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario $scenario --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --workflows=BeamSpotByRun,BeamSpotByLumi,SiStripQuality"
     runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario $scenario --lfn=/store/whatever --global-tag GLOBALTAG --skims SiStripCalZeroBias,SiStripCalMinBias,PromptCalibProd"
     runTest "${LOCAL_TEST_DIR}/RunDQMHarvesting.py --scenario $scenario --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG"
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBiasHI+SiStripCalMinBias --PhysicsSkim=DiJet+Photon+ZEE+ZMM"
done


declare -a arr=("AlCaLumiPixels" "AlCaTestEnable" "cosmicsEra_Run2_2017" "hcalnzsEra_Run2_2017" "ppEra_Run2_2017_trackingLowPU" "ppEra_Run2_2017_pp_on_XeXe" "ppEra_Run2_2017_ppRef" "ppEra_Run2_2017" "cosmicsEra_Run2_2018" "hcalnzsEra_Run2_2018" "ppEra_Run2_2018" "hcalnzsEra_Run2_2018_highBetaStar" "hcalnzsEra_Run2_2018_pp_on_AA" "ppEra_Run2_2018_highBetaStar" "ppEra_Run2_2018_pp_on_AA" "cosmicsHybridEra_Run2_2018" "cosmicsEra_Run3" "hcalnzsEra_Run3" "ppEra_Run3")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias"
done

declare -a arr=("AlCaLumiPixels" "cosmicsEra_Run2_2017" "ppEra_Run2_2017_trackingLowPU" "ppEra_Run2_2017_pp_on_XeXe" "ppEra_Run2_2017_ppRef" "ppEra_Run2_2017" "cosmicsEra_Run2_2018" "ppEra_Run2_2018" "ppEra_Run2_2018_highBetaStar" "ppEra_Run2_2018_pp_on_AA" "cosmicsHybridEra_Run2_2018" "cosmicsEra_Run3" "ppEra_Run3")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario $scenario --lfn=/store/whatever --global-tag GLOBALTAG --skims SiStripCalZeroBias,SiStripCalMinBias,PromptCalibProd"
     runTest "${LOCAL_TEST_DIR}/RunDQMHarvesting.py --scenario $scenario --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG"
done

declare -a arr=("ppEra_Run2_2017" "ppEra_Run2_2017_trackingLowPU" "ppEra_Run2_2017_pp_on_XeXe" "ppEra_Run2_2017_ppRef" "ppEra_Run2_2018" "ppEra_Run2_2018_highBetaStar" "ppEra_Run2_2018_pp_on_AA" "ppEra_Run3")
for scenario in "${arr[@]}"
do
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --miniaod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias"
     runTest "${LOCAL_TEST_DIR}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias --PhysicsSkim=@SingleMuon"
done

runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario AlCaTestEnable --global-tag GLOBALTAG --lfn /store/whatever --alcareco PromptCalibProdEcalPedestals "
runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario AlCaTestEnable --lfn=/store/whatever --global-tag GLOBALTAG --skims PromptCalibProdEcalPedestals"
runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario AlCaTestEnable --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --alcapromptdataset=PromptCalibProdEcalPedestals"

runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario AlCaLumiPixels --global-tag GLOBALTAG --lfn /store/whatever --alcareco AlCaPCCRandom+PromptCalibProdLumiPCC"
runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario AlCaLumiPixels --lfn=/store/whatever --global-tag GLOBALTAG --skims AlCaPCCRandom,PromptCalibProdLumiPCC"
runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario AlCaLumiPixels --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --alcapromptdataset=PromptCalibProdLumiPCC"

declare -a arr=("trackingOnlyEra_Run2_2017" "trackingOnlyEra_Run2_2018" "trackingOnlyEra_Run2_2018_highBetaStar" "trackingOnlyEra_Run2_2018_pp_on_AA" "trackingOnlyEra_Run3")
for scenario in "${arr[@]}"
do
    runTest "${LOCAL_TEST_DIR}/RunExpressProcessing.py --scenario $scenario --global-tag GLOBALTAG  --lfn /store/whatever  --alcarecos=TkAlMinBias+PromptCalibProdBeamSpotHP"
    runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario $scenario --lfn /store/whatever --global-tag GLOBALTAG --skims TkAlMinBias,PromptCalibProdBeamSpotHP"
    runTest "${LOCAL_TEST_DIR}/RunAlcaSkimming.py --scenario $scenario --lfn /store/whatever --global-tag GLOBALTAG --skims TkAlMinBias,PromptCalibProdBeamSpotHPLowPU"
    runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario $scenario --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --alcapromptdataset=PromptCalibProdBeamSpotHP"
    runTest "${LOCAL_TEST_DIR}/RunAlcaHarvesting.py --scenario $scenario --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --alcapromptdataset=PromptCalibProdBeamSpotHPLowPU"
done
