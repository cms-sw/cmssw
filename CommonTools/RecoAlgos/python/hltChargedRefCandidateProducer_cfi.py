import FWCore.ParameterSet.Config as cms

hltChargedRefCandidateProducer = cms.EDProducer("ChargedRefCandidateProducer",
    src          = cms.InputTag('tracks'),
    particleType = cms.string('pi+')
)
# foo bar baz
# kcnq4j2KE4wkK
