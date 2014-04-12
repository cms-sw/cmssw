import FWCore.ParameterSet.Config as cms

from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# The same as above but with a misaligned tracker geometry
misalignedGeomSearchTracker = cms.ESProducer("TrackerRecoGeometryESProducer",
    trackerGeometryLabel = cms.untracked.string('MisAligned'),
    appendToDataLabel = cms.string('MisAligned')
)


