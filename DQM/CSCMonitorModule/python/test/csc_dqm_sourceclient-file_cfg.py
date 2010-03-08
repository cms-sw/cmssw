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

    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/FEEB9DB8-7329-DF11-BB49-001D09F253FC.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/FCF75699-7F29-DF11-802D-001D09F291D7.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/FA924CE7-7029-DF11-868A-0019DB2F3F9A.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/F0C5F51D-7C29-DF11-B12C-0030487CD704.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/EE79F489-7D29-DF11-836E-0030487CD7B4.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/EE1A5B72-6F29-DF11-938F-001D09F24DA8.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/EC5B65D9-6929-DF11-987B-0030487CD180.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/E81E4D36-6B29-DF11-9110-001617C3B6E2.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/E2A23CE8-6429-DF11-B5EB-001617E30F50.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/E2454CE9-8329-DF11-9C51-0030487A3C92.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/DE4E1F39-6B29-DF11-9BE6-0030487A3C92.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/DA1B28AD-7A29-DF11-83BA-000423D6B48C.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/D6BE6A60-6D29-DF11-A511-000423D6A6F4.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/CAB036EF-7229-DF11-866E-0019B9F72BAA.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/C4DA26E0-7729-DF11-B5C6-0030487C8CB6.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/BE21BBDB-8329-DF11-ACDC-0030487C608C.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/B48D5BFB-7029-DF11-B0CF-000423D99E46.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/B0F975DC-6229-DF11-A577-0030487CD710.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/AE87C3C0-8129-DF11-97DC-0030487A18F2.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/AC3CA50A-6729-DF11-AA54-0030486780A8.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/A8E2A844-6429-DF11-952B-0030487CD17C.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/A298D2D8-6929-DF11-A962-001D09F25109.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/9C4A835C-6129-DF11-B326-001617C3B6E2.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/9638F115-7529-DF11-B567-0030487CD77E.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/9618F88A-7D29-DF11-894A-0030487C90C2.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/8E91CCA6-6C29-DF11-B852-001D09F29619.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/8E645094-7829-DF11-945A-001D09F252E9.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/8CE8F270-6829-DF11-8E95-001D09F2527B.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/8814400C-6729-DF11-9BA4-0030486780B8.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/86514DF4-7229-DF11-94AA-001D09F24FEC.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/7A75AEFA-6B29-DF11-A618-00304879FA4C.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/74A70CDB-8329-DF11-92CA-0030487A18D8.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/7407BB64-8729-DF11-89EE-001D09F2AF96.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/7270F464-8729-DF11-86DF-001D09F29146.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/6EB6C7EC-8529-DF11-B619-000423D99614.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/6CFCCA08-6E29-DF11-9F8E-001617C3B706.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/6A88F9F1-8529-DF11-B43E-001D09F250AF.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/6425A86A-9529-DF11-BCBF-0019B9F581C9.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/5C32211F-7C29-DF11-B93A-0030487CD7E0.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/5AC8F814-7529-DF11-9A1F-0030487CD6D2.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/4AF113C0-8129-DF11-B883-0030487A17B8.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/40DC63E4-7729-DF11-A2BF-0030487C90EE.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/4060F389-7D29-DF11-954C-00304879EDEA.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/3CF88999-7F29-DF11-95C8-001D09F24664.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/3A6CE564-8729-DF11-9AAF-001D09F2423B.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/3A37DB7D-7629-DF11-9F33-0030487CD812.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/2E4469B2-6729-DF11-B2F7-000423D94E1C.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/28856A31-7029-DF11-9D76-001617DC1F70.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/186DB116-6229-DF11-B151-000423D99160.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/105B729E-6529-DF11-A924-000423D6B358.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/0ED7BC94-7829-DF11-BB00-001D09F29538.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/06A61340-6429-DF11-AB5C-0030487C6090.root',
    '/store/data/Commissioning10/Cosmics/RAW/v3/000/130/269/0254F0EC-8529-DF11-9E85-001D09F23A34.root' 

#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/371/8CDF17C6-F9EA-DE11-ACD7-0030486730C6.root'
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/FE7337DE-25EA-DE11-A2D0-001D09F28EC1.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/EE33041A-36EA-DE11-AE1E-001D09F23A20.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/BE36112B-31EA-DE11-8312-001D09F251BD.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/B87C1EAD-34EA-DE11-BE92-000423D6CA72.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/B47E9285-2BEA-DE11-B2ED-001D09F292D1.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/B43A33F5-27EA-DE11-9B1C-000423D9517C.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/A2F900F5-20EA-DE11-B5EE-001D09F2905B.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/90F765FF-2EEA-DE11-9025-000423D98DD4.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/8850C466-1DEA-DE11-A3B6-001D09F241B9.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/7EED522F-38EA-DE11-A2E9-001D09F276CF.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/7E4E0046-27EA-DE11-8CAC-000423D944F0.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/625E8805-23EA-DE11-82BD-000423D992A4.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/56828F3A-33EA-DE11-BE7F-000423D98950.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/389CC283-1FEA-DE11-8EC3-001D09F253FC.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/2258D4EB-2CEA-DE11-9C46-001D09F2423B.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/181D4D94-2DEA-DE11-AFC8-000423D98634.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/16A1DED1-31EA-DE11-9782-001617C3B79A.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/1673780E-2AEA-DE11-95F5-003048D374F2.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/1267B25F-22EA-DE11-81F7-0030486730C6.root',
#       '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/124/294/08FADE70-24EA-DE11-8F7E-001617C3B6CE.root'
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


