import FWCore.ParameterSet.Config as cms

genParticles = cms.EDProducer("GenParticleCandidate2GenParticleProducer",
    src = cms.InputTag("genParticleCandidates")
)


# foo bar baz
# H7ObAQM45RM86
# 4qQe7Qy2HvqlY
