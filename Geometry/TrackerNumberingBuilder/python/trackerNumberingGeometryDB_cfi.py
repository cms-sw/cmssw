import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the Tracker Numbering.
#
TrackerGeometricDetESModule = cms.ESProducer("TrackerGeometricDetESModule",
    fromDDD = cms.bool(False),
    layerNumberPXB = cms.uint32(16),
    totalBlade = cms.uint32(24)
)


