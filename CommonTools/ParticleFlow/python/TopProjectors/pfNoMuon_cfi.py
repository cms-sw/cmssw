import FWCore.ParameterSet.Config as cms

pfNoMuon = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    enable =  cms.bool( True ),
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("noMuon"),
    topCollection = cms.InputTag("pfIsolatedMuons"),
    bottomCollection = cms.InputTag("pfNoPileUp"),
)

pfNoMuonJME = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    enable =  cms.bool( True ),
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("noMuon"),
    topCollection = cms.InputTag("pfIsolatedMuons"),
    bottomCollection = cms.InputTag("pfNoPileUpJME")
)
