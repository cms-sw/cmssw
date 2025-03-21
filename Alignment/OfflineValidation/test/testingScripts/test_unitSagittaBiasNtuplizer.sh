#! /bin/bash
function die { echo $1: status $2 ; exit $2; }

echo "TESTING SagittaBiasNtuplizer Analyser with RECO input..."
cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/SagittaBiasNtuplizer_cfg.py || die "Failure running SagittaBiasNtuplizer_cfg.py (with RECO input)" $?

echo "TESTING SagittaBiasNtuplizer Analyser with ALCARECO input..."

cmsRun ${CMSSW_BASE}/src/Alignment/OfflineValidation/test/SagittaBiasNtuplizer_cfg.py globalTag=141X_mcRun4_realistic_v3 fromRECO=False  myfile=/store/relval/CMSSW_15_0_0/RelValZMM_14/ALCARECO/TkAlDiMuonAndVertex-141X_mcRun4_realistic_v3_STD_RecoOnly_Run4D110_PU-v1/2580000/3aeb786a-439e-43b9-b1d6-aaf57831ddce.root || die "Failure running SagittaBiasNtuplizer_cfg.py (with ALCARECO input)" $?

