import FWCore.ParameterSet.Config as cms

hltTrackRefsForJetsBeforeSorting = cms.EDProducer("ChargedRefCandidateProducer",
    particleType = cms.string('pi+'),
    src = cms.InputTag("hltTrackWithVertexRefSelectorBeforeSorting")
)
