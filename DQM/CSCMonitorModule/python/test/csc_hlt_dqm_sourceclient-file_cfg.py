import FWCore.ParameterSet.Config as cms

process = cms.Process("CSC HLT DQM")

#-------------------------------------------------
# DQM Module Configuration
#-------------------------------------------------

process.load("DQM.CSCMonitorModule.test.csc_hlt_dqm_sourceclient_cfi")

#----------------------------
# Event Source
#-----------------------------

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source("NewEventStreamFileReader",
  fileNames = cms.untracked.vstring(
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0001.A.storageManager.0.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0001.A.storageManager.1.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0001.A.storageManager.2.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0001.A.storageManager.3.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0002.A.storageManager.0.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0002.A.storageManager.1.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0002.A.storageManager.2.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0002.A.storageManager.3.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0003.A.storageManager.0.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0003.A.storageManager.1.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0003.A.storageManager.2.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0003.A.storageManager.3.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0004.A.storageManager.0.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0004.A.storageManager.1.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0004.A.storageManager.2.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0004.A.storageManager.3.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0005.A.storageManager.0.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0005.A.storageManager.1.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0005.A.storageManager.2.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0005.A.storageManager.3.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0006.A.storageManager.0.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0006.A.storageManager.1.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0006.A.storageManager.2.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0006.A.storageManager.3.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0007.A.storageManager.0.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0007.A.storageManager.1.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0007.A.storageManager.2.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0007.A.storageManager.3.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0008.A.storageManager.0.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0008.A.storageManager.1.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0008.A.storageManager.2.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0008.A.storageManager.3.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0009.A.storageManager.0.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0009.A.storageManager.1.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0009.A.storageManager.1.0001.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0009.A.storageManager.2.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0009.A.storageManager.3.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0009.A.storageManager.3.0001.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0010.A.storageManager.1.0000.dat',
    'file:/tmp/valdo/GlobalCruzet4MW36.00061169.0010.A.storageManager.3.0000.dat',
  )
)

#process.source = cms.Source("PoolSource",
#  fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/CRUZET4_v1/Cosmics/RECO/CRZT210_V1_CSCSkim_trial_v1/0000/EC61736B-5873-DD11-9580-001A92971AA4.root'),
# fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/CRUZET3/Cosmics/RAW/v4/000/051/552/524A4381-4255-DD11-8FD6-001617E30F4C.root'),
# fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/CRUZET2/Cosmics/RAW/v1/000/046/873/04D28BCA-1F39-DD11-A6C1-001617C3B65A.root'),
#  debugVebosity = cms.untracked.uint32(1),
#  debugFlag = cms.untracked.bool(1)
#)

#-------------------------------------------------
# Global Tag
#-------------------------------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "sqlite_file:/nfshome0/malgeri/public/globtag/CRZT210_V1H.db"
#process.GlobalTag.connect = "frontier://FrontierDev/CMS_COND_CSC"
process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRZT210_V1H::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#----------------------------
# DQM Environment
#-----------------------------

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
# DQM Playback Environment
#-----------------------------

process.load("DQM.Integration.test.environment_playback_cfi")
process.dqmEnv.subSystemFolder    = "CSC"

process.DQM.collectorHost = 'pccmsdqm02.cern.ch'
#process.DQM.collectorHost = 'localhost'
process.dqmSaver.dirName = '.'

#--------------------------
# Message Logger
#--------------------------

MessageLogger = cms.Service("MessageLogger",

  suppressInfo = cms.untracked.vstring('source'),
  suppressDebug = cms.untracked.vstring('source'),
  suppressWarning = cms.untracked.vstring('source'),

  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    WARNING = cms.untracked.PSet(
      limit = cms.untracked.int32(0)
    ),
    noLineBreaks = cms.untracked.bool(False)
  ),

  detailedInfo = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO')
  ),

  critical = cms.untracked.PSet(
    threshold = cms.untracked.string('ERROR')
  ),

  debug = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG')
  ),

  debugModules = cms.untracked.vstring('CSCHLTMonitormodule'),

  destinations = cms.untracked.vstring(
#    'debug',
#    'detailedInfo', 
#    'critical', 
#    'cout'
  )

)

#--------------------------
# Sequences
#--------------------------

process.p = cms.Path(process.dqmCSCClient+process.dqmEnv+process.dqmSaver)


