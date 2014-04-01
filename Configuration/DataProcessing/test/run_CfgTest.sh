#!/bin/bash

# Test suite for various ConfigDP scenarios
# run using: scram build runtest 
# feel free to contribute with your favourite configuration


# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

function runTest { echo $1 ; python $1 || die "Failure for configuration: $1" $?; }


INPUT=${LOCAL_TEST_DIR}/RunExpressProcessing.py

runTest "${INPUT} --scenario cosmics --global-tag GLOBALTAG::ALL --lfn /store/whatever --fevt --alca --dqm"
runTest "${INPUT} --scenario pp --global-tag GLOBALTAG::ALL --lfn /store/whatever --fevt --alca --dqm"
runTest "${INPUT} --scenario HeavyIons --global-tag GLOBALTAG::ALL --lfn /store/whatever --fevt --alca --dqm"


INPUT=${LOCAL_TEST_DIR}/RunRepack.py

runTest "${INPUT} --select-events HLT:path1,HLT:path2 --lfn /store/whatever"


INPUT=${LOCAL_TEST_DIR}/RunPromptReco.py

runTest "${INPUT} --scenario=cosmics --reco --aod --alcareco --dqm --global-tag GLOBALTAG::ALL --lfn=/store/whatever"
runTest "${INPUT} --scenario=pp --reco --aod --alcareco --dqm --global-tag GLOBALTAG::ALL --lfn=/store/whatever"
runTest "${INPUT} --scenario=HeavyIons --reco --aod --alcareco --dqm --global-tag GLOBALTAG::ALL --lfn=/store/whatever"
runTest "${INPUT} --scenario=AlCaLumiPixels --reco --aod --alcareco --dqm --global-tag GLOBALTAG::ALL --lfn=/store/whatever"
#runTest "${INPUT} --scenario=AlCaP0 --reco --aod --alcareco --dqm --global-tag GLOBALTAG::ALL --lfn=/store/whatever"
#runTest "${INPUT} --scenario=AlCaPhiSymEcal --reco --aod --alcareco --dqm --global-tag GLOBALTAG::ALL --lfn=/store/whatever"
runTest "${INPUT} --scenario=AlCaTestEnable --reco --aod --alcareco --dqm --global-tag GLOBALTAG::ALL --lfn=/store/whatever"
runTest "${INPUT} --scenario=hcalnzs --reco --aod --alcareco --dqm --global-tag GLOBALTAG::ALL --lfn=/store/whatever"

INPUT=${LOCAL_TEST_DIR}/RunAlcaSkimming.py

runTest "${INPUT} --scenario pp --lfn=/store/whatever --global-tag GLOBALTAG::ALL --skims SiStripCalZeroBias,SiStripCalMinBias,TkAlMinBias,PromptCalibProd"
runTest "${INPUT} --scenario cosmics --lfn /store/whatever --global-tag GLOBALTAG::ALL --skims SiStripCalZeroBias,SiStripPCLHistos"
runTest "${INPUT} --scenario HeavyIons --lfn=/store/whatever --global-tag GLOBALTAG::ALL --skims SiStripCalZeroBias,SiStripCalMinBias,TkAlMinBiasHI,PromptCalibProd"
runTest "${INPUT} --scenario AlCaLumiPixels --lfn /store/whatever --global-tag GLOBALTAG::ALL --skims LumiPixels"
#runTest "${INPUT} --scenario AlCaP0 --lfn /store/whatever --global-tag GLOBALTAG::ALL --skims EcalCalPi0Calib"
#runTest "${INPUT} --scenario AlCaPhiSymEcal --lfn /store/whatever --global-tag GLOBALTAG::ALL --skims EcalCalPhiSym"


INPUT=${LOCAL_TEST_DIR}/RunAlcaHarvesting.py

runTest "${INPUT} --scenario pp --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG::ALL --workflows=BeamSpotByRun,BeamSpotByLumi,SiStripQuality"
runTest "${INPUT} --scenario cosmics --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG::ALL --workflows=SiStripQuality"
runTest "${INPUT} --scenario HeavyIons --lfn /store/whatever --dataset /A/B/C --global-tag GLOBALTAG::ALL --workflows=BeamSpotByRun,BeamSpotByLumi,SiStripQuality"

INPUT=${LOCAL_TEST_DIR}/RunDQMHarvesting.py

runTest "${INPUT} --scenario pp --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG::ALL"
runTest "${INPUT} --scenario cosmics --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG::ALL"
runTest "${INPUT} --scenario AlCaLumiPixels --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG::ALL"
#runTest "${INPUT} --scenario AlCaP0 --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG::ALL"
#runTest "${INPUT} --scenario AlCaPhiSymEcal --lfn /store/whatever --run 12345 --dataset /A/B/C --global-tag GLOBALTAG::ALL"
