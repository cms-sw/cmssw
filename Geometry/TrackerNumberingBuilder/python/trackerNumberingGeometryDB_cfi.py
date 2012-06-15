import FWCore.ParameterSet.Config as cms
import Geometry.TrackerNumberingBuilder.pixelGeometryConstants_cfi as pixelGeometryConstants_cfi

#
# This cfi should be included to build the Tracker Numbering.
#
TrackerGeometricDetESModule = cms.ESProducer("TrackerGeometricDetESModule",
    fromDDD = cms.bool(False),
    pixelGeometryConstants = cms.PSet(pixelGeometryConstants_cfi.pixelGeometryConstants)
)


