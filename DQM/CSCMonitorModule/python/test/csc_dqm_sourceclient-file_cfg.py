import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDQM")

#-------------------------------------------------
# DQM Module Configuration
#-------------------------------------------------

process.load("DQM.CSCMonitorModule.csc_dqm_sourceclient_cfi")
process.load("DQM.CSCMonitorModule.csc_daq_info_cfi")
process.load("DQM.CSCMonitorModule.csc_dcs_info_cfi")
process.load("DQM.CSCMonitorModule.csc_certification_info_cfi")

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
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/371/8CDF17C6-F9EA-DE11-ACD7-0030486730C6.root'
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/FE7337DE-25EA-DE11-A2D0-001D09F28EC1.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/EE33041A-36EA-DE11-AE1E-001D09F23A20.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/BE36112B-31EA-DE11-8312-001D09F251BD.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/B87C1EAD-34EA-DE11-BE92-000423D6CA72.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/B47E9285-2BEA-DE11-B2ED-001D09F292D1.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/B43A33F5-27EA-DE11-9B1C-000423D9517C.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/A2F900F5-20EA-DE11-B5EE-001D09F2905B.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/90F765FF-2EEA-DE11-9025-000423D98DD4.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/8850C466-1DEA-DE11-A3B6-001D09F241B9.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/7EED522F-38EA-DE11-A2E9-001D09F276CF.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/7E4E0046-27EA-DE11-8CAC-000423D944F0.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/625E8805-23EA-DE11-82BD-000423D992A4.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/56828F3A-33EA-DE11-BE7F-000423D98950.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/389CC283-1FEA-DE11-8EC3-001D09F253FC.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/2258D4EB-2CEA-DE11-9C46-001D09F2423B.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/181D4D94-2DEA-DE11-AFC8-000423D98634.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/16A1DED1-31EA-DE11-9782-001617C3B79A.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/1673780E-2AEA-DE11-95F5-003048D374F2.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/1267B25F-22EA-DE11-81F7-0030486730C6.root',
       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/08FADE70-24EA-DE11-8F7E-001617C3B6CE.root'
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

process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder    = "CSC"

process.DQM.collectorPort = 9190
process.DQM.collectorHost = 'cms-uflap03.dyndns.cern.ch'
process.dqmSaver.convention = "Online"
process.dqmSaver.dirName = "/tmp/valdo"
process.dqmSaver.producer = "DQM"

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
process.GlobalTag.globaltag = 'GR09_31X_V1P::All' 
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
    WARNING = cms.untracked.PSet(
      limit = cms.untracked.int32(0)
    ),
    noLineBreaks = cms.untracked.bool(False)
  ),

  detailedInfo = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG')
  ),

  critical = cms.untracked.PSet(
    threshold = cms.untracked.string('ERROR')
  ),

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

process.p = cms.Path(process.dqmCSCClient * process.cscDaqInfo * process.cscDcsInfo * process.cscCertificationInfo + process.dqmEnv + process.dqmSaver)
#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscMonitor * process.dqmCSCClient + process.dqmEnv + process.dqmSaver)


