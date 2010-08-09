import FWCore.ParameterSet.Config as cms

L3TrackCandCombiner = cms.EDProducer("L3TrackCandCombiner",
    labels = cms.VInputTag()
)
