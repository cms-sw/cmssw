#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python3 $1 || die "Failure for configuration: $1" $?; }

declare -a arr=("l1ScoutingEra_Run3_2026")
for scenario in "${arr[@]}"
do
     # RECO, AOD and DQMIO are not supported for L1-Scouting scenarios, and RunPromptReco.py is expected to ignore them without failing
     runTest "${SCRAM_TEST_PATH}/RunPromptReco.py --scenario $scenario --nanoaod --global-tag GLOBALTAG --lfn=/store/whatever --nanoFlavours=@L1Scout --reco --aod --dqmio"
     runTest "${SCRAM_TEST_PATH}/RunPromptReco.py --scenario $scenario --nanoaod --global-tag GLOBALTAG --lfn=/store/whatever --nanoFlavours=@L1Scout"
     runTest "${SCRAM_TEST_PATH}/RunPromptReco.py --scenario $scenario --nanoaod --global-tag GLOBALTAG --lfn=/store/whatever --nanoFlavours=@L1ScoutSelect"
done
