import FWCore.ParameterSet.Config as cms
packedGenParticles = cms.EDProducer("PATPackedGenParticleProducer",
    inputCollection = cms.InputTag("genParticles"),
    map = cms.InputTag("genParticles"),
    inputOriginal = cms.InputTag("genParticles"),
    inputVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    maxRapidity = cms.double(6)
)
