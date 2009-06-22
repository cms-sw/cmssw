import FWCore.ParameterSet.Config as cms

noElectron = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("noElectron"),
    topCollection = cms.InputTag("isolatedElectrons"),
    bottomCollection = cms.InputTag("noMuon"),
)
