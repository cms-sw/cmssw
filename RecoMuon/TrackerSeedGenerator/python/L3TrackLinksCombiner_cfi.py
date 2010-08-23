import FWCore.ParameterSet.Config as cms

L3TrackLinksCombiner = cms.EDProducer(
    "L3TrackLinksCombiner",
    labels = cms.VInputTag(
    cms.InputTag("hltL3MuonsOIState"),
    cms.InputTag("hltL3MuonsOIHit"),
    cms.InputTag("hltL3MuonsIOHit")
    )
    )
