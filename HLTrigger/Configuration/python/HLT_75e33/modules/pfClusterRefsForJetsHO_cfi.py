import FWCore.ParameterSet.Config as cms

pfClusterRefsForJetsHO = cms.EDProducer("PFClusterRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("particleFlowClusterHO")
)
# foo bar baz
# YceAZmFhlwHhQ
# exBuvnyetwbs7
