#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python3 $1 || die "Failure for configuration: $1" $?; }

runTest "${SCRAM_TEST_PATH}/RunExpressProcessing.py --scenario AlCaLumiPixels_Run3 --global-tag GLOBALTAG --lfn /store/whatever --alcareco AlCaPCCRandom+PromptCalibProdLumiPCC"
runTest "${SCRAM_TEST_PATH}/RunAlcaSkimming.py --scenario AlCaLumiPixels_Run3 --lfn=/store/whatever --global-tag GLOBALTAG --skims AlCaPCCRandom+PromptCalibProdLumiPCC"
runTest "${SCRAM_TEST_PATH}/RunAlcaHarvesting.py --scenario AlCaLumiPixels_Run3 --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG --alcapromptdataset=PromptCalibProdLumiPCC"

