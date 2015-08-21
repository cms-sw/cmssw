import FWCore.ParameterSet.Config as cms
from Geometry.HcalEventSetup.HcalRelabel_cfi import HcalReLabel

HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP" ,
                                        appendToDataLabel = cms.string("_master"),
                                        UseOldLoader = cms.bool(False),
                                        HcalReLabel = HcalReLabel
                                        )

HcalGeometryFromDBEP = cms.ESProducer("HcalGeometryFromDBEP",
                                      applyAlignment = cms.bool(False)
                                      )

HcalGeometryToDBEP   = cms.ESProducer("HcalGeometryToDBEP" ,
                                      applyAlignment = cms.bool(False) ,
                                      appendToDataLabel = cms.string("_toDB")
                                      )

