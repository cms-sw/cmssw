#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python3 $1 || die "Failure for configuration: $1" $?; }

declare -a arr=("hltScoutingEra_Run3_2024")
for scenario in "${arr[@]}"
do
     runTest "${SCRAM_TEST_PATH}/RunPromptReco.py --scenario $scenario --nanoaod --global-tag GLOBALTAG --lfn=/store/whatever --nanoFlavours=@Scout"
done
