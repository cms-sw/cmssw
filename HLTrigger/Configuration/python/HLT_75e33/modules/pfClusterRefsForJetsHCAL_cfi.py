import FWCore.ParameterSet.Config as cms

pfClusterRefsForJetsHCAL = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("hltParticleFlowClusterHCAL")
)
# foo bar baz
# uWyjxOtfme7lO
