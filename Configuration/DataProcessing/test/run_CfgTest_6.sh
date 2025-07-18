#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python3 $1 || die "Failure for configuration: $1" $?; }

declare -a arr=("AlCaLumiPixels" "cosmicsEra_Run2_2018" "ppEra_Run2_2018" "ppEra_Run2_2018_highBetaStar" "ppEra_Run2_2018_pp_on_AA" "cosmicsHybridEra_Run2_2018" "cosmicsEra_Run3" "ppEra_Run3" "AlCaLumiPixels_Run3")
for scenario in "${arr[@]}"
do
     runTest "${SCRAM_TEST_PATH}/RunAlcaSkimming.py --scenario $scenario --lfn=/store/whatever --global-tag GLOBALTAG --skims SiStripCalZeroBias+SiStripCalMinBias+PromptCalibProd"
     runTest "${SCRAM_TEST_PATH}/RunDQMHarvesting.py --scenario $scenario --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG"
done

