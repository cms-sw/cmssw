import FWCore.ParameterSet.Config as cms
from Geometry.HcalEventSetup.HcalRelabel_cfi import HcalReLabel

process = cms.Process("HcalGeometryTest")

process.load("Geometry.HcalCommonData.testPhase0GeometryXML_cfi")
process.load("Geometry.HcalCommonData.hcalDDConstants_cff")
process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")

process.HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP" ,
                                                appendToDataLabel = cms.string("_master"),
                                                HcalReLabel = HcalReLabel
                                                )
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hga = cms.EDAnalyzer("HcalGeometryDetIdAnalyzer",
                             HcalReLabel = HcalReLabel,
                             HCALGeometryLabel = cms.string("_master") )

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hga)
