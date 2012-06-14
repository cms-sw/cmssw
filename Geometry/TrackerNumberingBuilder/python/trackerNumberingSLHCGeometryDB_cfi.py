import FWCore.ParameterSet.Config as cms
import Geometry.TrackerNumberingBuilder.pixelSLHCGeometryConstants_cfi as pixelSLHCGeometryConstants_cfi

#
# This cfi should be included to build the Tracker Numbering for SLHC.
#
TrackerGeometricDetESModule = cms.ESProducer("TrackerGeometricDetESModule",
    fromDDD = cms.bool(False),
    pixelGeometryConstants = cms.PSet(pixelSLHCGeometryConstants_cfi.pixelSLHCGeometryConstants)
)


