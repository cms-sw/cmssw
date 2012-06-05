import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the Tracker Numbering for SLHC.
#
TrackerGeometricDetESModule = cms.ESProducer("TrackerGeometricDetESModule",
    fromDDD = cms.bool(True),
    layerNumberPXB = cms.uint32(18),
    totalBlade = cms.uint32(56)
)


