
import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#from xml text files #process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

#from db xml source: #process.load("GeometryReaders.XMLIdealGeometryESSource.cmsGeometryDB_cff")
#from db sql file:   #process.load("Geometry.CaloEventSetup.xmlsqlitefile")
#from db frontier:   #process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

#xml source: #process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

#reco source from db: #
process.load("Geometry.CaloEventSetup.CaloGeometryDBReader_cfi")

#reco from db sql     : #process.load("Geometry.CaloEventSetup.calodbsqlitefile")
#reco from db frontier: #
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
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

process.p1 = cms.Path(process.etta*process.ctgw*process.cga*process.mfa)


