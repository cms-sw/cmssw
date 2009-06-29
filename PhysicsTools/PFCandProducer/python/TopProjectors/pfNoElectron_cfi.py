import FWCore.ParameterSet.Config as cms

pfNoElectron = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("noElectron"),
    topCollection = cms.InputTag("pfIsolatedElectrons"),
    bottomCollection = cms.InputTag("pfNoMuon"),
)
