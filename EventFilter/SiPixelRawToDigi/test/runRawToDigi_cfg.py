import FWCore.ParameterSet.Config as cms

process = cms.Process("MyRawToDigi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Services_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100))

process.source = cms.Source("PoolSource",
# fileNames =  cms.untracked.vstring('file:rawdata.root')
fileNames =  cms.untracked.vstring(
  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/205/217/2EF61B7D-F216-E211-98C3-001D09F28D54.root",
#  "rfio:/castor/cern.ch/cms/store/data/Run2012D/MinimumBias/RAW/v1/000/208/686/A88F66A0-393F-E211-9287-002481E0D524.root",
#    "file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100_71_pre7/raw/raw2.root"
 )
)

# Cabling
#  include "CalibTracker/Configuration/data/Tracker_FakeConditions.cff"
#process.load("CalibTracker.Configuration.SiPixel_FakeConditions_cff")
#process.load("CalibTracker.Configuration.SiPixelCabling.SiPixelCabling_SQLite_cff")
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRAFT_V3P::All"
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
#process.siPixelCabling.connect = 'sqlite_file:cabling.db'
#process.siPixelCabling.toGet = cms.VPSet(cms.PSet(
#    record = cms.string('SiPixelFedCablingMapRcd'),
#    tag = cms.string('SiPixelFedCablingMap_v14')
#))


# Choose the global tag here:
#process.GlobalTag.globaltag = "GR_P_V40::All"
# for data in V7
# process.GlobalTag.globaltag = "GR_R_71_V1::All"
# for MC 
process.GlobalTag.globaltag = "MC_71_V1::All"

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
# for simultaions 
process.siPixelDigis.InputLabel = 'siPixelRawData'
# for data
#process.siPixelDigis.InputLabel = 'source'
#process.siPixelDigis.InputLabel = 'rawDataCollector'
process.siPixelDigis.IncludeErrors = True
process.siPixelDigis.Timing = False 
#process.siPixelDigis.UseCablingTree = True 

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelDigis'),
    destinations = cms.untracked.vstring('r2d'),
    r2d = cms.untracked.PSet( threshold = cms.untracked.string('WARNING'))
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName =  cms.untracked.string('file:digis.root'),
#    fileName =  cms.untracked.string('file:/afs/cern.ch/work/d/dkotlins/public/data/digis/digis_1k.root'),
    outputCommands = cms.untracked.vstring("drop *","keep *_siPixelDigis_*_*")
)

process.p = cms.Path(process.siPixelDigis)
process.ep = cms.EndPath(process.out)
