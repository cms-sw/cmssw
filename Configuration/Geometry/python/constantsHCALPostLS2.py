import FWCore.ParameterSet.Config as cms

import Geometry.HcalEventSetup.hcalSLHCTopologyConstants_cfi as hcalTopologyConstants_cfi

def customise(process):

    process.hcalTopologyIdeal.hcalTopologyConstants.mode = cms.string('HcalTopologyMode::SLHC')
    process.hcalTopologyIdeal.hcalTopologyConstants.maxDepthHB = cms.int32(7)
    process.hcalTopologyIdeal.hcalTopologyConstants.maxDepthHE = cms.int32(7)

    process.hcalTopologyIdeal.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
    process.es_hardcode.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
    process.CaloTowerHardcodeGeometryEP = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)

    return process
