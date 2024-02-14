#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python3 $1 || die "Failure for configuration: $1" $?; }

declare -a arr=("ppEra_Run3" "ppEra_Run3_2023" "ppEra_Run3_2023_repacked")
for scenario in "${arr[@]}"
do
     runTest "${SCRAM_TEST_PATH}/RunPromptReco.py --scenario $scenario --reco --aod --miniaod --nanoaod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias"
     runTest "${SCRAM_TEST_PATH}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever --alcareco TkAlMinBias+SiStripCalMinBias --PhysicsSkim=@Muon0"
done

