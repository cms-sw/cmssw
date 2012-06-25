import FWCore.ParameterSet.Config as cms
from Geometry.HcalEventSetup.HcalRelabel_cfi import HcalReLabel

HcalHardcodeGeometryEP = cms.ESProducer( "HcalHardcodeGeometryEP" ,
                                         appendToDataLabel = cms.string("_master"),
                                         HcalReLabel = HcalReLabel
                                         )

HcalGeometryToDBEP = cms.ESProducer( "HcalGeometryToDBEP" ,
                                     applyAlignment = cms.bool(False) ,
                                     appendToDataLabel = cms.string("_toDB"),
                                     HcalReLabel = HcalReLabel
                                     )

