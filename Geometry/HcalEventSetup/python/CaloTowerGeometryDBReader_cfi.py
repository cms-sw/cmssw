import FWCore.ParameterSet.Config as cms

CaloTowerGeometryFromDBEP = cms.ESProducer( "CaloTowerGeometryFromDBEP",
                                            applyAlignment = cms.bool(False)
                                            )

import Geometry.HcalEventSetup.hcalTopologyConstants_cfi as hcalTopologyConstants_cfi
CaloTowerGeometryFromDBEP.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)
