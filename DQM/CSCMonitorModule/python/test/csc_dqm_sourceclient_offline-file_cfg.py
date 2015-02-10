import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDQM")

#-------------------------------------------------
# DQM Module Configuration
#-------------------------------------------------

process.load("DQM.CSCMonitorModule.csc_dqm_sourceclient_offline_cff")
#process.load("DQMServices.Components.MEtoEDMConverter_cff")
#process.load("DQM.CSCMonitorModule.csc_daq_info_cfi")
#process.load("DQM.CSCMonitorModule.csc_dcs_info_cfi")
#process.load("DQM.CSCMonitorModule.csc_certification_info_cfi")

#-------------------------------------------------
# Offline DQM Module Configuration
#-------------------------------------------------

process.load("DQMOffline.Muon.CSCMonitor_cfi")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.csc2DRecHits.readBadChambers = cms.bool(False)

#----------------------------
# Event Source
#-----------------------------

#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))
process.source = cms.Source("PoolSource",
  fileNames  = cms.untracked.vstring(
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/D65FC93B-5BAB-DF11-A9AF-001D09F2516D.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/C492E26A-51AB-DF11-8F03-003048F118C2.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/A6519D80-53AB-DF11-87EB-001D09F25109.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/9E418993-4EAB-DF11-901B-0030487C7E18.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/90C67078-4CAB-DF11-BAE1-0030487CD184.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/8802AD8E-61AB-DF11-B775-0030487A17B8.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/84CABB05-5EAB-DF11-9B05-0030487A3DE0.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/82ABB08A-5AAB-DF11-B543-003048F11C58.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/5ACA6051-56AB-DF11-83EA-0030487CD178.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/5632EBC2-4BAB-DF11-AC22-001617C3B77C.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/249B981E-52AB-DF11-96F0-003048F024FE.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/08F7838E-61AB-DF11-89DD-0030487CD180.root',
    '/store/data/Run2010A/Cosmics/RAW/v1/000/143/227/0827F81F-59AB-DF11-80E0-003048F024FE.root'
  ),
  #skipEvents = cms.untracked.uint32(1129)
)

#----------------------------
# DQM Environment
#-----------------------------

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#process.DQMStore.referenceFileName = '/home/dqmdevlocal/reference/csc_reference.root'
process.DQMStore.referenceFileName = '/afs/cern.ch/user/v/valdo/data/csc_reference.root'
#process.DQMStore.referenceFileName = '/nfshome0/valdo/CMSSW_2_1_0/src/DQM/CSCMonitorModule/data/csc_reference.root'

#----------------------------
# DQM Playback Environment
#-----------------------------

process.load("DQM.Integration.test.environment_playback_cfi")
process.dqmEnv.subSystemFolder    = "CSC"

process.DQM.collectorHost = 'pccmsdqm02.cern.ch'
#process.DQM.collectorHost = 'localhost'
process.dqmSaver.dirName = '/tmp/valdo'

#-----------------------------
# Magnetic Field
#-----------------------------

process.load("Configuration/StandardSequences/MagneticField_cff")

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------------------
# Global Tag
#-------------------------------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
#process.GlobalTag.connect = "sqlite_file:/nfshome0/malgeri/public/globtag/CRZT210_V1H.db"
#process.GlobalTag.connect = "frontier://FrontierDev/CMS_COND_CSC"
#process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRZT210_V1H::All"
#process.GlobalTag.globaltag = 'CRAFT_V3P::All'
#process.GlobalTag.globaltag = "CRAFT_30X::All"
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
#process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRAFT_V17H::All"
#process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG"
#process.GlobalTag.globaltag = 'GR09_31X_V1H::All' 
#process.GlobalTag.globaltag = 'GR09_31X_V1P::All' 
process.GlobalTag.globaltag = 'GR10_P_V2::All' 
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


#--------------------------
# Web Service
#--------------------------

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")
process.AdaptorConfig = cms.Service("AdaptorConfig")

#--------------------------
# Message Logger
#--------------------------

MessageLogger = cms.Service("MessageLogger",

# suppressInfo = cms.untracked.vstring('source'),
# suppressInfo = cms.untracked.vstring('*'),

  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
#    WARNING = cms.untracked.PSet(
#      limit = cms.untracked.int32(0)
#    ),
#    noLineBreaks = cms.untracked.bool(False)
  ),

  detailedInfo = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG')
  ),

#  critical = cms.untracked.PSet(
#    threshold = cms.untracked.string('ERROR')
#  ),

  debugModules = cms.untracked.vstring('*'),

  destinations = cms.untracked.vstring(
    'detailedInfo', 
    'critical', 
    'cout'
  )

)

#--------------------------
# Sequences
#--------------------------

process.p = cms.Path(
  process.dqmCSCClient + 
  process.dqmEnv + 
  process.dqmSaver) 
#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscMonitor * process.dqmCSCClient + process.dqmEnv + process.dqmSaver)


