import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDQM")

#-------------------------------------------------
# DQM Module Configuration
#-------------------------------------------------

process.load("DQM.CSCMonitorModule.test.csc_dqm_sourceclient_cfi")
process.load("DQM.CSCMonitorModule.test.csc_daq_info_cfi")
process.load("DQM.CSCMonitorModule.test.csc_dcs_info_cfi")
process.load("DQM.CSCMonitorModule.test.csc_certification_info_cfi")

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

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source("PoolSource",
    fileNames  = cms.untracked.vstring(
      #'/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/422F78CA-7019-DE11-A599-001617E30CD4.root',
      #'/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/764D08CA-7019-DE11-813F-001617C3B69C.root',
      #'/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/963C5DCA-7019-DE11-9ABF-001617DBD316.root',
      #'/store/data/Commissioning09/Cosmics/RAW/v1/000/079/035/C882B9D5-7219-DE11-8B69-000423D6BA18.root'
      '/store/data/Commissioning08/Cosmics/RAW/v1/000/066/910/8CA64FCF-259F-DD11-B86D-000423D99BF2.root'
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
process.dqmSaver.dirName = '.'

#-----------------------------
# Magnetic Field
#-----------------------------

process.load("Configuration/StandardSequences/MagneticField_cff")

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.Geometry_cff")

#-------------------------------------------------
# Global Tag
#-------------------------------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "sqlite_file:/nfshome0/malgeri/public/globtag/CRZT210_V1H.db"
#process.GlobalTag.connect = "frontier://FrontierDev/CMS_COND_CSC"
#process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRZT210_V1H::All"
#process.GlobalTag.globaltag = 'CRAFT_V3P::All'
process.GlobalTag.globaltag = "CRAFT_30X::All"
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

process.p = cms.Path(process.dqmCSCClient * process.cscDaqInfo + process.cscDcsInfo * process.cscCertificationInfo + process.dqmEnv + process.dqmSaver)
#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscMonitor * process.dqmCSCClient + process.dqmEnv + process.dqmSaver)


