import FWCore.ParameterSet.Config as cms
from Geometry.HcalEventSetup.HcalRelabel_cfi import HcalReLabel

process = cms.Process("HcalGeometryTest")

#process.load("Configuration.Geometry.GeometryExtendedPostLS2_cff")
process.load("Geometry.HcalCommonData.testPhase2GeometryXML_cfi")
#process.load("Geometry.CMSCommonData.cmsExtendedGeometry2023HGCalXML_cfi")
process.load("Geometry.HcalCommonData.hcalSimNumberingInitialization_cfi")
process.load("Geometry.HcalCommonData.hcalRecNumberingInitialization_cfi")
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
                             HCALGeometryLabel = cms.string("_master") )

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hga)
