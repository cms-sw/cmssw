import FWCore.ParameterSet.Config as cms

L3TrackCandCombiner = cms.EDProducer(
    "L3TrackCandCombiner",
    labels = cms.VInputTag(
    cms.InputTag("hltL3TrackCandidateFromL2IOHit"),
    cms.InputTag("hltL3TrackCandidateFromL2OIHit"),
    cms.InputTag("hltL3TrackCandidateFromL2OIState"),
    )
    )
