import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.Geometry.GeometryExtended_cff")
process.load('Configuration.Geometry.GeometryExtendedReco_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(4) )

process.source = cms.Source("EmptySource")

process.etta = cms.EDAnalyzer("dumpEcalTrigTowerMapping")

process.ctgw = cms.EDAnalyzer("testEcalGetWindow")

process.cga = cms.EDAnalyzer("CaloGeometryAnalyzer",
                             fullEcalDump = cms.untracked.bool(True)
                             )

process.mfa = cms.EDAnalyzer("testMagneticField")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('calogeom.root')
                                   )

process.p1 = cms.Path(process.etta*process.ctgw*process.cga*process.mfa)


