
import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#process.load('Configuration/StandardSequences/GeometryDB_cff')

process.load('GeometryReaders.XMLIdealGeometryESSource.cmsGeometryDB_cff')

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = 'MC_36Y_V4::All'
process.source = cms.Source("EmptySource")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(4) )

process.source = cms.Source("EmptySource")

process.etta = cms.EDAnalyzer("dumpEcalTrigTowerMapping")

process.ctgw = cms.EDAnalyzer("testEcalGetWindow")

process.cga = cms.EDAnalyzer("CaloGeometryAnalyzer",
                             fullEcalDump = cms.untracked.bool(True)
                             )

process.mfa = cms.EDAnalyzer("testMagneticField")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('calogeom.root')
                                   )

process.p1 = cms.Path(process.etta*process.ctgw*process.cga*process.mfa)


