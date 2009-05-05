
import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.CaloEventSetup.calodbsqlitefile")

#process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

#process.load("Geometry.CaloEventSetup.AlignedCaloGeometry_cfi")

#process.load("Geometry.CaloEventSetup.FakeCaloAlignments_cff")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(4)
)

process.source = cms.Source("EmptySource")

process.etta = cms.EDFilter("dumpEcalTrigTowerMapping")

process.ctgw = cms.EDFilter("testEcalGetWindow")

process.cga = cms.EDAnalyzer("CaloGeometryAnalyzer",
    fullEcalDump = cms.untracked.bool(True)
)

process.mfa = cms.EDFilter("testMagneticField")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.TFileService = cms.Service("TFileService", fileName = cms.string('calogeom.root') )

process.p1 = cms.Path(process.etta*process.ctgw*process.cga*process.mfa)


