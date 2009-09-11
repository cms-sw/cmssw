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
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/F6BEA315-5994-DE11-87DE-003048D37514.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/F691794F-5D94-DE11-9B57-000423D6B42C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/F651264A-2D94-DE11-94AB-000423D6C8EE.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/F4FF9109-2994-DE11-984F-001617E30F50.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/F43C5672-1E94-DE11-9DAD-000423D98C20.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/EEB04B2D-3294-DE11-94AA-0016177CA7A0.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/EC35B4BF-6D94-DE11-842C-001D09F291D7.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/E80E5F3D-3994-DE11-8802-000423D6B42C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/E807F225-6094-DE11-9143-003048D2C1C4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/E6B32F50-5894-DE11-9038-000423D9939C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/E4F4A481-3194-DE11-9812-000423D6C8EE.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/DEF7C4F2-3994-DE11-B8B6-000423D6C8E6.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/D6F9D0F9-4594-DE11-B335-000423D6A6F4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/D03FE1C1-3C94-DE11-9662-003048D37514.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/CCCF7590-2594-DE11-984A-000423D9863C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/CA920CE7-2694-DE11-8CDF-000423D6CA72.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/CA4A2721-5B94-DE11-A7DB-000423D6CA72.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/C80587E1-4F94-DE11-B6F8-000423D6CA42.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/C29519FA-5194-DE11-8C9A-000423D6B48C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/C086584E-3494-DE11-A26C-000423D9870C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/BEA0B006-4D94-DE11-A03E-000423D6B42C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/BE579AD9-1F94-DE11-A3E0-000423D9989E.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/B410E154-4094-DE11-A207-000423D98DD4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/B2596617-4894-DE11-B167-003048D37580.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/AA503E86-4494-DE11-9A94-003048D3750A.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/A83282B9-2994-DE11-B79C-000423D6B5C4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/A69343B3-4194-DE11-B633-000423D6AF24.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/9E5DF658-3B94-DE11-9034-000423D9870C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/9E1064A2-3A94-DE11-AB0E-000423D6C8EE.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/9CE52DE4-5694-DE11-85EE-000423D6CA6E.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/9ACCF417-5494-DE11-9D0A-000423D6AF24.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/98D4F4BB-6394-DE11-9E9C-000423D6AF24.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/98AB2772-4E94-DE11-BCD5-003048D3750A.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/965F4830-4A94-DE11-BFF3-000423D6A6F4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/94C0182B-3E94-DE11-9986-001617E30CD4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/94B7852A-6294-DE11-B033-000423D986A8.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/949D655C-4794-DE11-A51D-000423D9870C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/90CEDF99-3F94-DE11-9851-000423DD2F34.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/8C058FB2-3594-DE11-91C6-000423D6C8EE.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/8AECC461-2F94-DE11-BFEA-000423D9863C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/8AAEABE7-3294-DE11-AE26-000423D6CA6E.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/84A2BCC9-6094-DE11-8ADE-001617E30F48.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/7CC9878A-3894-DE11-A372-000423D986A8.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/7C434450-2894-DE11-8591-001617C3B6DE.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/7A56CD21-2B94-DE11-AFB7-000423D6B5C4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/7A17CFE4-4A94-DE11-966A-001617C3B6CC.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/6C5F4D08-2994-DE11-A86C-001617E30D40.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/681C6A14-2494-DE11-86CB-000423D6B42C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/66BA38CC-2494-DE11-8718-000423D6B48C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/6206EE93-5094-DE11-A965-000423D6CA6E.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/5AC94E31-5694-DE11-B702-000423D6CA72.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/589DBD78-5594-DE11-8186-000423D6B42C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/52F52FB0-5E94-DE11-B404-003048D37456.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/4E05CECB-4394-DE11-B4DC-000423D6A6F4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/4C56857E-4994-DE11-B24B-000423D985E4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/4885B000-3594-DE11-B2D6-000423D6CA72.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/462D0CC6-2294-DE11-9A5B-001617E30F4C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/44C52556-4C94-DE11-A5E7-003048D2C1C4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/3E662871-2A94-DE11-AB22-000423D6B42C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/3AB7A712-3094-DE11-A319-000423D6B5C4.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/3A59F8BE-5994-DE11-BDAB-003048D37560.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/3A3FDFAD-5294-DE11-9DFD-000423D6AF24.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/24DEAD41-2194-DE11-98BB-003048D374F2.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/1ED90E2E-3E94-DE11-81B1-0019DB29C614.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/1A931FAE-2E94-DE11-862C-000423D9939C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/16B07DAC-5E94-DE11-89C6-003048D2BF1C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/1069FE1B-4394-DE11-A09C-000423D9939C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/082D3D92-2C94-DE11-AE1F-000423D9863C.root',
    '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/389/065358D4-3794-DE11-9A25-000423D6C8E6.root'
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

process.p = cms.Path(process.dqmCSCClient * process.cscDaqInfo * process.cscDcsInfo * process.cscCertificationInfo + process.dqmEnv + process.dqmSaver)
#process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscMonitor * process.dqmCSCClient + process.dqmEnv + process.dqmSaver)


