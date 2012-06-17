import FWCore.ParameterSet.Config as cms
from Geometry.HcalEventSetup.HcalRelabel_cfi import HcalReLabel

HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP" ,
                                        HcalReLabel = HcalReLabel
)
