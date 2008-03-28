import FWCore.ParameterSet.Config as cms

from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
#The same, but as above for misaligned Tracker geometry
misalignedTrackerGeometry = cms.ESProducer("TrackerDigiGeometryESModule",
    appendToDataLabel = cms.string('MisAligned'),
    fromDDD = cms.bool(True),
    applyAlignment = cms.untracked.bool(False)
)


