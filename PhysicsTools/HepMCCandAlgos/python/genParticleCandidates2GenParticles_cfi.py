import FWCore.ParameterSet.Config as cms

genParticles = cms.EDProducer("GenParticleCandidate2GenParticleProducer",
    src = cms.InputTag("genParticleCandidates")
)


