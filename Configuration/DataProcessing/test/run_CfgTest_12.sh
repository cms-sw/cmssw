#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python3 $1 || die "Failure for configuration: $1" $?; }

declare -a arr=("trackingOnlyEra_Run2_2018" "trackingOnlyEra_Run2_2018_highBetaStar" "trackingOnlyEra_Run2_2018_pp_on_AA" "trackingOnlyEra_Run3" "trackingOnlyEra_Run3_pp_on_PbPb")
for scenario in "${arr[@]}"
do
    runTest "${SCRAM_TEST_PATH}/RunExpressProcessing.py --scenario $scenario --global-tag GLOBALTAG  --lfn /store/whatever  --alcarecos=TkAlMinBias+PromptCalibProdBeamSpotHP"
    runTest "${SCRAM_TEST_PATH}/RunAlcaSkimming.py --scenario $scenario --lfn /store/whatever --global-tag GLOBALTAG --skims TkAlMinBias+PromptCalibProdBeamSpotHP"
    runTest "${SCRAM_TEST_PATH}/RunAlcaSkimming.py --scenario $scenario --lfn /store/whatever --global-tag GLOBALTAG --skims TkAlMinBias+PromptCalibProdBeamSpotHPLowPU"
    runTest "${SCRAM_TEST_PATH}/RunAlcaHarvesting.py --scenario $scenario --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --alcapromptdataset=PromptCalibProdBeamSpotHP"
    runTest "${SCRAM_TEST_PATH}/RunAlcaHarvesting.py --scenario $scenario --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --alcapromptdataset=PromptCalibProdBeamSpotHPLowPU"
done
