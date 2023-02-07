#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING inspect ALCARECO data ..."
cmsRun ${LOCAL_TEST_DIR}/inspectData_cfg.py unitTest=True trackCollection=ALCARECOTkAlCosmicsCTF0T || die "Failure running inspectData_cfg.py" $?

echo "TESTING G4e refitter ..."
cmsRun ${LOCAL_TEST_DIR}/testG4Refitter_cfg.py maxEvents=10  || die "Failure running testG4Refitter_cfg.py" $?

echo "TESTING Pixel BaryCenter Analyser ..."
cmsRun ${LOCAL_TEST_DIR}/PixelBaryCentreAnalyzer_cfg.py unitTest=True || die "Failure running PixelBaryCentreAnalyzer_cfg.py" $?

echo "TESTING CosmicTrackSplitting Analyser ..."
cmsRun  ${LOCAL_TEST_DIR}/testSplitterValidation_cfg.py unitTest=True || die "Failure running testSplitterValidation_cfg.py" $?
