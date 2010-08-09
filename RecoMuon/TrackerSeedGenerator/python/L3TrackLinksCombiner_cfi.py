import FWCore.ParameterSet.Config as cms

L3TrackLinksCombiner = cms.EDProducer("L3TrackLinksCombiner",
    labels = cms.VInputTag()
)
