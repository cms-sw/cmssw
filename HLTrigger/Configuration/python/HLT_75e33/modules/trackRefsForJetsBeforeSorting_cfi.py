import FWCore.ParameterSet.Config as cms

trackRefsForJetsBeforeSorting = cms.EDProducer("ChargedRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("trackWithVertexRefSelectorBeforeSorting")
)
# foo bar baz
# TxgzAz7A1AJrW
# vPmKQsVSb8vym
