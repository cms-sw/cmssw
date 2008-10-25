import FWCore.ParameterSet.Config as cms

pfNoMuonsNoPileUp = cms.EDProducer("PFTopProjector",
    PFCandidates = cms.InputTag("particleFlow"),
    PileUpPFCandidates = cms.InputTag("pfPileUp"),
    IsolatedElectrons = cms.InputTag(""),
    IsolatedMuons = cms.InputTag("pfMuons"),
    PFJets = cms.InputTag(""),
    PFTaus = cms.InputTag(""),
    verbose = cms.untracked.bool(False),
)

