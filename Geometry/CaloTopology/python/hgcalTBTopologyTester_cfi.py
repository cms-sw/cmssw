import FWCore.ParameterSet.Config as cms

from Geometry.CaloTopology.hgcalTBTopologyTesterEE_cfi import *

hgcalTBTopologyTesterHEF = hgcalTBTopologyTesterEE.clone(
    detectorName = cms.string("HGCalHESiliconSensitive"),
    types        = cms.vint32(0,0,1,1,1,1,2,2,2),
    layers       = cms.vint32(1,2,9,3,4,5,6,7,8),
    sector       = cms.vint32(2,3,3,4,5,6,7,8,9),
    cells        = cms.vint32(0,10,15,8,8,10,10,15,15))
