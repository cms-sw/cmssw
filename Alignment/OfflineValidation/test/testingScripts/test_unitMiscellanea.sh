#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING G4e refitter ..."
cmsRun ${LOCAL_TEST_DIR}/testG4Refitter_cfg.py maxEvents=50  || die "Failure running testG4Refitter_cfg.py" $?

echo "TESTING Pixel BaryCenter Analyser ..."
cmsRun ${LOCAL_TEST_DIR}/PixelBaryCentreAnalyzer_cfg.py unitTest=True || die "Failure running PixelBaryCentreAnalyzer_cfg.py" $?
