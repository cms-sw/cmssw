import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process("GeometryTest", Run3_dd4hep)


process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')

process.load('FWCore.MessageLogger.MessageLogger_cfi')
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.CaloGeom=dict()

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(4) )

process.source = cms.Source("EmptySource")

process.etta = cms.EDAnalyzer("DumpEcalTrigTowerMapping")

process.ctgw = cms.EDAnalyzer("TestEcalGetWindow")

process.cga = cms.EDAnalyzer("CaloGeometryAnalyzer",
                             fullEcalDump = cms.untracked.bool(True)
                             )

process.mfa = cms.EDAnalyzer("testMagneticField")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('calogeom.root')
                                   )

# process.p1 = cms.Path(process.etta*process.ctgw*process.cga*process.mfa)
process.p1 = cms.Path(process.etta*process.ctgw*process.cga)
# FIXME Restore magnetic field test. Code has to be added to read field record


