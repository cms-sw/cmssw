import FWCore.ParameterSet.Config as cms

HcalGeometryFromDBEP = cms.ESProducer("HcalGeometryFromDBEP",
                                      applyAlignment = cms.bool(False)
                                      )
import Geometry.HcalEventSetup.hcalTopologyConstants_cfi as hcalTopologyConstants_cfi
HcalGeometryFromDBEP.hcalTopologyConstants = cms.PSet(hcalTopologyConstants_cfi.hcalTopologyConstants)

HcalAlignmentEP = cms.ESProducer("HcalAlignmentEP")

