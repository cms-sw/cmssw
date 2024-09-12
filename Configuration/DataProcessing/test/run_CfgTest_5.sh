#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtests
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python3 $1 || die "Failure for configuration: $1" $?; }

declare -a arr=("AlCaLumiPixels" "AlCaTestEnable" "cosmicsEra_Run2_2018" "hcalnzsEra_Run2_2018" "ppEra_Run2_2018" "hcalnzsEra_Run2_2018_highBetaStar" "hcalnzsEra_Run2_2018_pp_on_AA" "ppEra_Run2_2018_highBetaStar" "ppEra_Run2_2018_pp_on_AA" "cosmicsHybridEra_Run2_2018" "cosmicsEra_Run3" "hcalnzsEra_Run3" "ppEra_Run3" "AlCaLumiPixels_Run3" "AlCaPhiSymEcal_Nano" "AlCaPPS_Run3" "ppEra_Run3_pp_on_PbPb" "hcalnzsEra_Run3_pp_on_PbPb" "ppEra_Run3_pp_on_PbPb_approxSiStripClusters" "ppEra_Run3_2023" "ppEra_Run3_pp_on_PbPb_2023" "ppEra_Run3_pp_on_PbPb_approxSiStripClusters_2023" "ppEra_Run3_2024" "ppEra_Run3_pp_on_PbPb_2024" "ppEra_Run3_pp_on_PbPb_approxSiStripClusters_2024" "ppEra_Run3_2024_UPC" "ppEra_Run3_2024_ppRef")
for scenario in "${arr[@]}"
do
     runTest "${SCRAM_TEST_PATH}/RunPromptReco.py --scenario $scenario --reco --aod --dqmio --global-tag GLOBALTAG --lfn=/store/whatever  --alcareco TkAlMinBias+SiStripCalMinBias"
done

