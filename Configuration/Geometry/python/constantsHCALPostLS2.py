import FWCore.ParameterSet.Config as cms

import Geometry.HcalEventSetup.hcalTopologyConstants_cfi

def customise(process):

    process.hcalTopologyIdeal.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
    process.es_hardcode.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
    process.CaloTowerHardcodeGeometryEP = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)

    return process
