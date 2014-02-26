import FWCore.ParameterSet.Config as cms

import Geometry.TrackerGeometryBuilder.trackerSLHCGeometryConstants_cfi as trackerGeometryConstants_cfi

def customise(process):

    process.trackerGeometry.trackerGeometryConstants = cms.PSet(trackerGeometryConstants_cfi.trackerGeometryConstants)
    process.idealForDigiTrackerGeometry.trackerGeometryConstants = cms.PSet(trackerGeometryConstants_cfi.trackerGeometryConstants)
    
    process.trackerNumberingGeometry.fromDDD = cms.bool( True )
    process.trackerNumberingGeometry.layerNumberPXB = cms.uint32(18)
    process.trackerNumberingGeometry.totalBlade = cms.uint32(56)

    return process
