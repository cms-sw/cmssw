import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQM.CSCMonitorModule.csc_hlt_dqm_sourceclient_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('CSCFileReader'),
    readerPset = cms.untracked.PSet(
        firstEvent = cms.untracked.int32(0),
        RUI01 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI01_Monitor_000.raw'),
        RUI02 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI02_Monitor_000.raw'),
        RUI03 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI03_Monitor_000.raw'),
        RUI04 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI04_Monitor_000.raw'),
        RUI05 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI05_Monitor_000.raw'),
        RUI06 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI06_Monitor_000.raw'),
        RUI07 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI07_Monitor_000.raw'),
        RUI08 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI08_Monitor_000.raw'),
        RUI09 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI09_Monitor_000.raw'),
        RUI10 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI10_Monitor_000.raw'),
        RUI11 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI11_Monitor_000.raw'),
        RUI12 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI12_Monitor_000.raw'),
        RUI13 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI13_Monitor_000.raw'),
        RUI14 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI14_Monitor_000.raw'),
        RUI15 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI15_Monitor_000.raw'),
        RUI16 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI16_Monitor_000.raw'),
        RUI17 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI17_Monitor_000.raw'),
        RUI18 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI18_Monitor_000.raw'),
        RUI19 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI19_Monitor_000.raw'),
        RUI20 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI20_Monitor_000.raw'),
        RUI21 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI21_Monitor_000.raw'),
        RUI22 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI22_Monitor_000.raw'),
        RUI23 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI23_Monitor_000.raw'),
        RUI24 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI24_Monitor_000.raw'),
        RUI25 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI25_Monitor_000.raw'),
        RUI26 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI26_Monitor_000.raw'),
        RUI27 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI27_Monitor_000.raw'),
        RUI28 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI28_Monitor_000.raw'),
        RUI29 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI29_Monitor_000.raw'),
        RUI30 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI30_Monitor_000.raw'),
        RUI31 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI31_Monitor_000.raw'),
        RUI32 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI32_Monitor_000.raw'),
        RUI33 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI33_Monitor_000.raw'),
        RUI34 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI34_Monitor_000.raw'),
        RUI35 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI35_Monitor_000.raw'),
        RUI36 = cms.untracked.vstring('/tmp/valdo/csc_00063201_EmuRUI36_Monitor_000.raw'),
        FED750 = cms.untracked.vstring('RUI01','RUI02','RUI03','RUI04','RUI05','RUI06','RUI07','RUI08','RUI09'),
        FED751 = cms.untracked.vstring('RUI10','RUI11','RUI12','RUI13','RUI14','RUI15','RUI16','RUI17','RUI18'),
        FED752 = cms.untracked.vstring('RUI19','RUI20','RUI21','RUI22','RUI23','RUI24','RUI25','RUI26','RUI27'),
        FED753 = cms.untracked.vstring('RUI28','RUI29','RUI30','RUI31','RUI32','RUI33','RUI34','RUI35','RUI36'),
    )
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

process.p = cms.Path(process.dqmCSCClient+process.dqmEnv+process.dqmSaver)

process.DQM.collectorHost = 'pccmsdqm02.cern.ch'

process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

process.dqmEnv.subSystemFolder = 'CSC'



