import FWCore.ParameterSet.Config as cms

pfTopProjection = cms.EDProducer("PFTopProjector",
    PFCandidates = cms.InputTag("particleFlow"),
    PileUpPFCandidates = cms.InputTag("pfPileUp"),
    IsolatedElectrons = cms.InputTag("pfElectrons"),
    IsolatedMuons = cms.InputTag("pfMuons"),
    PFJets = cms.InputTag("pfJets"),
    PFTaus = cms.InputTag("allLayer0Taus"),
    verbose = cms.untracked.bool(False)
)
