import FWCore.ParameterSet.Config as cms

pfNoPileUp = cms.EDProducer("PFTopProjector",
    PFCandidates = cms.InputTag("particleFlow"),
    PileUpPFCandidates = cms.InputTag("pfPileUp"),
    IsolatedElectrons = cms.InputTag(""),
    IsolatedMuons = cms.InputTag(""),
    PFJets = cms.InputTag(""),
    PFTaus = cms.InputTag(""),
    verbose = cms.untracked.bool(False),
)

