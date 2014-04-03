import FWCore.ParameterSet.Config as cms
packedGenCandidates = cms.EDProducer("PATPackedGenCandidateProducer",
    inputCollection = cms.InputTag("genParticles"),
    map = cms.InputTag("genParticles"),
    inputVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    maxEta = cms.double(5)
)
