import FWCore.ParameterSet.Config as cms

l1tHPSPFTauProducer = cms.EDProducer("L1HPSPFTauProducer",
    HW = cms.bool(True),
    debug = cms.bool(False),
    mightGet = cms.optional.untracked.vstring,
    nTaus = cms.int32(16),
    srcL1PFCands = cms.InputTag("l1tLayer1","Puppi"),
    srcL1PFJets = cms.InputTag("l1tPhase1JetCalibrator9x9trimmed","Phase1L1TJetFromPfCandidates"),
    useJets = cms.bool(False)
)
