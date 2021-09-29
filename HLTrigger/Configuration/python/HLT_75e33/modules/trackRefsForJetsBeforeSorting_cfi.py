import FWCore.ParameterSet.Config as cms

trackRefsForJetsBeforeSorting = cms.EDProducer("ChargedRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("trackWithVertexRefSelectorBeforeSorting")
)
