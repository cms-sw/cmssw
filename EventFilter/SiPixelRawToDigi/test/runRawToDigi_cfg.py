import FWCore.ParameterSet.Config as cms

process = cms.Process("RawToDigi1")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100))

process.source = cms.Source("PoolSource", fileNames =  cms.untracked.vstring('file:rawdata.root'))

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")

# Cabling
#  include "CalibTracker/Configuration/data/Tracker_FakeConditions.cff"
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRAFT_V3P::All"
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
process.load("CalibTracker.Configuration.SiPixel_FakeConditions_cff")
process.load("CalibTracker.Configuration.SiPixelCabling.SiPixelCabling_SQLite_cff")
process.siPixelCabling.connect = 'sqlite_file:cabling.db'
process.siPixelCabling.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelFedCablingMapRcd'),
    tag = cms.string('SiPixelFedCablingMap_v14')
))

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = 'siPixelRawData'
process.siPixelDigis.IncludeErrors = True
process.siPixelDigis.Timing = True 
process.siPixelDigis.UseCablingTree = True 

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelDigis'),
    destinations = cms.untracked.vstring('r2d'),
    r2d = cms.untracked.PSet( threshold = cms.untracked.string('ERROR'))
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName =  cms.untracked.string('file:digis.root'),
    outputCommands = cms.untracked.vstring("drop *","keep *_siPixelDigis_*_*")
)

process.p = cms.Path(process.siPixelDigis)
process.ep = cms.EndPath(process.out)
