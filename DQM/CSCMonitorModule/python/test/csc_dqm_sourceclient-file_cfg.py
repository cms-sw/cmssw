import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCDQM")

#-------------------------------------------------
# DQM Module Configuration
#-------------------------------------------------

process.load("DQM.CSCMonitorModule.test.csc_dqm_sourceclient_cfi")

#-------------------------------------------------
# Offline DQM Module Configuration
#-------------------------------------------------

process.load("DQMOffline.Muon.CSCMonitor_cfi")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.csc2DRecHits.readBadChambers = cms.bool(False)

#----------------------------
# Event Source
#-----------------------------

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(50))
process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('CSCFileReader'),
    readerPset = cms.untracked.PSet(
        firstEvent = cms.untracked.int32(0),
        RUI05 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI05_Default_000_080618_201111_UTC.raw'),
        RUI04 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI04_Default_000_080618_201111_UTC.raw'),
        RUI07 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI07_Default_000_080618_201111_UTC.raw'),
        RUI06 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI06_Default_000_080618_201111_UTC.raw'),
        RUI01 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI01_Default_000_080618_201111_UTC.raw'),
        RUI26 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI26_Default_000_080618_201111_UTC.raw'),
        RUI03 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI03_Default_000_080618_201111_UTC.raw'),
        RUI02 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI02_Default_000_080618_201111_UTC.raw'),
        RUI29 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI29_Default_000_080618_201111_UTC.raw'),
        RUI28 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI28_Default_000_080618_201111_UTC.raw'),
        RUI09 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI09_Default_000_080618_201111_UTC.raw'),
        RUI08 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI08_Default_000_080618_201111_UTC.raw'),
        RUI20 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI20_Default_000_080618_201111_UTC.raw'),
        RUI27 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI27_Default_000_080618_201111_UTC.raw'),
        RUI25 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI25_Default_000_080618_201111_UTC.raw'),
        RUI24 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI24_Default_000_080618_201111_UTC.raw'),
        RUI21 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI21_Default_000_080618_201111_UTC.raw'),
        RUI36 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI36_Default_000_080618_201111_UTC.raw'),
        RUI16 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI16_Default_000_080618_201111_UTC.raw'),
        RUI17 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI17_Default_000_080618_201111_UTC.raw'),
        RUI14 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI14_Default_000_080618_201111_UTC.raw'),
        RUI15 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI15_Default_000_080618_201111_UTC.raw'),
        RUI12 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI12_Default_000_080618_201111_UTC.raw'),
        RUI13 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI13_Default_000_080618_201111_UTC.raw'),
        RUI10 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI10_Default_000_080618_201111_UTC.raw'),
        RUI11 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI11_Default_000_080618_201111_UTC.raw'),
        RUI34 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI34_Default_000_080618_201111_UTC.raw'),
        RUI35 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI35_Default_000_080618_201111_UTC.raw'),
        RUI23 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI23_Default_000_080618_201111_UTC.raw'),
        RUI30 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI30_Default_000_080618_201111_UTC.raw'),
        RUI31 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI31_Default_000_080618_201111_UTC.raw'),
        RUI18 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI18_Default_000_080618_201111_UTC.raw'),
        RUI19 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI19_Default_000_080618_201111_UTC.raw'),
        RUI32 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI32_Default_000_080618_201111_UTC.raw'),
        RUI33 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI33_Default_000_080618_201111_UTC.raw'),
        RUI22 = cms.untracked.vstring('/tmp/valdo/csc_00000000_EmuRUI22_Default_000_080618_201111_UTC.raw'),
        FED750 = cms.untracked.vstring('RUI01', 
            'RUI02', 
            'RUI03', 
            'RUI04', 
            'RUI05', 
            'RUI06', 
            'RUI07', 
            'RUI08', 
            'RUI09', 
            'RUI10', 
            'RUI11', 
            'RUI12', 
            'RUI13', 
            'RUI14', 
            'RUI15', 
            'RUI16', 
            'RUI17', 
            'RUI18', 
            'RUI19', 
            'RUI20', 
            'RUI21', 
            'RUI22', 
            'RUI23', 
            'RUI24', 
            'RUI25', 
            'RUI26', 
            'RUI27', 
            'RUI28', 
            'RUI29', 
            'RUI30', 
            'RUI31', 
            'RUI32', 
            'RUI33', 
            'RUI34', 
            'RUI35', 
            'RUI36')
    )
)

#----------------------------
# DQM Environment
#-----------------------------

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#process.DQMStore.referenceFileName = '/home/dqmdevlocal/reference/csc_reference.root'
#process.DQMStore.referenceFileName = '/afs/cern.ch/user/v/valdo/CMSSW_2_1_0/src/DQM/CSCMonitorModule/data/csc_reference.root'
process.DQMStore.referenceFileName = '/nfshome0/valdo/CMSSW_2_1_0/src/DQM/CSCMonitorModule/data/csc_reference.root'

#----------------------------
# DQM Playback Environment
#-----------------------------

process.load("DQM.Integration.test.environment_playback_cfi")
process.dqmEnv.subSystemFolder    = "CSC"

#process.DQM.collectorHost = 'pccmsdqm02.cern.ch'
process.DQM.collectorHost = 'localhost'
process.dqmSaver.dirName = '.'

#-----------------------------
# Magnetic Field
#-----------------------------

process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

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
process.GlobalTag.connect ="frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRZT210_V1H::All"
process.GlobalTag.globaltag = 'CRAFT_V3P::All'
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
  suppressInfo = cms.untracked.vstring('*'),

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

  debugModules = cms.untracked.vstring('CSCMonitormodule'),

#  destinations = cms.untracked.vstring('detailedInfo', 
#    'critical', 
#    'cout')

)

#--------------------------
# Sequences
#--------------------------

#process.p = cms.Path(process.dqmClient + process.dqmEnv + process.dqmSaver)
process.p = cms.Path(process.muonCSCDigis * process.csc2DRecHits * process.cscSegments * process.cscMonitor * process.dqmClient + process.dqmEnv + process.dqmSaver)


