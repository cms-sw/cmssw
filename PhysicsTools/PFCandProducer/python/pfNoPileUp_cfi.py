import FWCore.ParameterSet.Config as cms

pfNoPileUp = cms.EDProducer("PFTopProjectorPF2PAT",
    PFCandidates = cms.InputTag("particleFlow"),
    PileUpPFCandidates = cms.InputTag("pfPileUp"),
    IsolatedElectrons = cms.InputTag(""),
    IsolatedMuons = cms.InputTag(""),
    PFJets = cms.InputTag(""),
    PFTaus = cms.InputTag(""),
    verbose = cms.untracked.bool(False),
)

