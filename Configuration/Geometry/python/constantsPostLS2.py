import FWCore.ParameterSet.Config as cms

import Geometry.HcalEventSetup.hcalSLHCTopologyConstants_cfi as hcalTopologyConstants_cfi
import Geometry.TrackerGeometryBuilder.trackerSLHCGeometryConstants_cfi as trackerGeometryConstants_cfi

def customise(process):

    process.trackerGeometry.trackerGeometryConstants = cms.PSet(trackerGeometryConstants_cfi.trackerGeometryConstants)
    process.idealForDigiTrackerGeometry.trackerGeometryConstants = cms.PSet(trackerGeometryConstants_cfi.trackerGeometryConstants)
    
    process.trackerNumberingGeometry.fromDDD = cms.bool( True )
    process.trackerNumberingGeometry.layerNumberPXB = cms.uint32(18)
    process.trackerNumberingGeometry.totalBlade = cms.uint32(56)

##     process.hcalTopologyIdeal.hcalTopologyConstants.mode = cms.string('HcalTopologyMode::SLHC')
##     process.hcalTopologyIdeal.hcalTopologyConstants.maxDepthHB = cms.int32(7)
##     process.hcalTopologyIdeal.hcalTopologyConstants.maxDepthHE = cms.int32(7)

    process.hcalTopologyIdeal.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
    process.es_hardcode.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
    process.CaloTowerHardcodeGeometryEP = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)

    return process
