import FWCore.ParameterSet.Config as cms
from Geometry.HcalEventSetup.HcalRelabel_cfi import HcalReLabel

process = cms.Process("HcalGeometryTest")

process.load("Geometry.HcalCommonData.testPhase0GeometryXML_cfi")
process.load("Geometry.HcalCommonData.hcalDDConstants_cff")
process.load("Geometry.HcalEventSetup.hcalTopologyIdeal_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    categories = cms.untracked.vstring('HCalGeom'),
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
      threshold = cms.untracked.string('DEBUG'),
      default = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
      ),
      HCalGeom = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
      )
    )
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.hga = cms.EDAnalyzer("HcalGeometryTester",
                             HCALGeometryLabel = cms.string("_master"),
                             UseOldLoader      = cms.bool(False))

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hga)
