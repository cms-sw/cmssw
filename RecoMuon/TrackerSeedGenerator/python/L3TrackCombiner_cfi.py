import FWCore.ParameterSet.Config as cms

L3TrackCombiner = cms.EDProducer(
    "L3TrackCombiner",
    labels = cms.VInputTag(
    cms.InputTag("hltL3MuonsOIState"),
    cms.InputTag("hltL3MuonsOIHit"),
    cms.InputTag("hltL3MuonsIOHit"),
    )
    )
