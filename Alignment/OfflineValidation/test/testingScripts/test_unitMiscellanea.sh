#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING inspect ALCARECO data ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/inspectData_cfg.py unitTest=True trackCollection=ALCARECOTkAlCosmicsCTF0T || die "Failure running inspectData_cfg.py" $?

echo "TESTING G4e refitter ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/testG4Refitter_cfg.py maxEvents=10  || die "Failure running testG4Refitter_cfg.py" $?

echo "TESTING Pixel BaryCenter Analyser ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/PixelBaryCentreAnalyzer_cfg.py unitTest=True || die "Failure running PixelBaryCentreAnalyzer_cfg.py" $?

echo "TESTING CosmicTrackSplitting Analyser ..."
cmsRun  ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/testSplitterValidation_cfg.py unitTest=True || die "Failure running testSplitterValidation_cfg.py" $?
