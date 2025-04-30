#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING inspect ALCARECO data ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/inspectData_cfg.py unitTest=True isCosmics=True trackCollection=ALCARECOTkAlCosmicsCTF0T || die "Failure running inspectData_cfg.py" $?

echo "TESTING inspect Phase2 ALCARECO data ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/inspectData_cfg.py unitTest=True isCosmics=False globalTag='' trackCollection=ALCARECOTkAlZMuMu isDiMuonData=True Detector='Run4D98' || die "Failure running inspectData_cfg.py on Phase-2 input" $?

echo "TESTING G4e refitter ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/testG4Refitter_cfg.py maxEvents=10  || die "Failure running testG4Refitter_cfg.py" $?

echo "TESTING Pixel BaryCenter Analyser ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/python/TkAlAllInOneTool/PixelBaryCentreAnalyzer_cfg.py unitTest=True || die "Failure running PixelBaryCentreAnalyzer_cfg.py" $?

echo "TESTING CosmicTrackSplitting Analyser ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/testSplitterValidation_cfg.py unitTest=True || die "Failure running testSplitterValidation_cfg.py" $?

echo "TESTING TkAlV0sAnalyzer Analyser ..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/TkAlV0sAnalyzer_cfg.py unitTest=True || die "Failure running TkAlV0sAnalyzer_cfg.py" $?
-- dummy change --
