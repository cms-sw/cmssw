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
process.source = cms.Source("PoolSource",
    fileNames  = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RAW/v1/000/066/910/8CA64FCF-259F-DD11-B86D-000423D99BF2.root'),
    skipEvents = cms.untracked.uint32(25900)
)

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

process.p = cms.Path(process.cscDQMEvF+process.dqmEnv+process.dqmSaver)

