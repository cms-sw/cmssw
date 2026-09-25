import FWCore.ParameterSet.Config as cms

hltTrackRefsForJetsBeforeSorting4D = cms.EDProducer("ChargedRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("hltTrackWithVertexRefSelectorBeforeSorting4D")
)
