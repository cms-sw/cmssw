import FWCore.ParameterSet.Config as cms
from Geometry.HcalEventSetup.HcalRelabel_cfi import HcalReLabel

HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP" ,
                                        UseOldLoader = cms.bool(False),
                                        HcalReLabel = HcalReLabel
)
