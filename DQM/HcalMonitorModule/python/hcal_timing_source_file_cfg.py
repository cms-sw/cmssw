import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalTimingTest")

process.MessageLogger = cms.Service("MessageLogger",
     categories   = cms.untracked.vstring(''),
     destinations = cms.untracked.vstring('cout'),
     debugModules = cms.untracked.vstring('*'),
     cout = cms.untracked.PSet(
         threshold = cms.untracked.string('WARNING'),
         WARNING = cms.untracked.PSet(limit = cms.untracked.int32(0))
     )
)

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff")
process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.l1GtUnpack.DaqGtInputTag = 'source'

process.load("L1Trigger.Configuration.L1DummyConfig_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("L1Trigger.L1ExtraFromDigis.l1extraParticles_cff")
process.load("HLTrigger.HLTfilters.hltLevel1GTSeed_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/00D22895-3109-DE11-A8A8-0030487A3C9A.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/02427DDC-4609-DE11-B6F4-001D09F26C5C.root',
 '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/04329D1E-4D09-DE11-9434-001617C3B78C.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/04710543-4A09-DE11-90DA-000423D174FE.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/06685B48-3C09-DE11-B465-001617C3B5E4.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/06818771-2D09-DE11-BB50-0019B9F72CC2.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/08C598DA-3009-DE11-8ABA-001617C3B6DC.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/08EDE497-3109-DE11-B76A-00304879FA4C.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/0CEE5190-4E09-DE11-8729-001617E30F48.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/0E866721-5909-DE11-84F7-001D09F231C9.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/101518B2-3609-DE11-A3C5-001D09F2A690.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/10D0DE5A-4809-DE11-BFCC-001D09F231C9.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/12209C80-3B09-DE11-9730-001D09F28C1E.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/12549884-3409-DE11-AC0B-0019B9F7310E.root',
  '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/12549884-3409-DE11-AC0B-0019B9F7310E.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/12924C9E-3709-DE11-8F2A-001D09F24EC0.root',
        '/store/data/Commissioning09/Cosmics/RAW/v1/000/077/023/145FCD88-3409-DE11-B6EC-001D09F24682.root'
    )
)

process.hcalunpacker = cms.EDProducer("HcalRawToDigi",
    FilterDataQuality = cms.bool(True),
    HcalFirstFED = cms.untracked.int32(700),
    InputLabel = cms.InputTag("rawDataCollector"),
 UnpackCalib = cms.untracked.bool(False),
    FEDs = cms.untracked.vint32(700, 701, 702, 703, 704,
        705, 706, 707, 708, 709,
        710, 711, 712, 713, 714,
        715, 716, 717, 718, 719,
        720, 721, 722, 723, 724,
        725, 726, 727, 728, 729,
        730, 731),
    lastSample = cms.int32(9),
    firstSample = cms.int32(0)
)
process.load("DQM.HcalMonitorModule.HcalTimingModule_cfi")

process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

process.es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('QIEShape',
        'QIEData',
        'ChannelQuality',
        'HcalQIEData',
        'Pedestals',
        'PedestalWidths',
        'Gains',
        'GainWidths',
        'ZSThresholds',
        'RespCorrs')
)

process.hcalConditions = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('HcalElectronicsMapRcd'),
        tag = cms.string('official_emap_v7.01')
    )),
    connect = cms.string('frontier://Frontier/CMS_COND_21X_HCAL')
)

### include to get DQM histogramming services
process.load("DQMServices.Core.DQM_cfg")

### set the verbose
process.DQMStore.verbose = 0


#### BEGIN DQM Online Environment #######################

### replace YourSubsystemName by the name of your source ###
### use it for dqmEnv, dqmSaver
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = 'localhost'
process.DQM.collectorPort = 9190
### path where to save the output file
process.dqmSaver.dirName = '.'
### the filename prefix
process.dqmSaver.producer = 'DQM'
### possible conventions are "Online", "Offline" and "RelVal"
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'HcalTest'

process.p = cms.Path(process.hcalunpacker*process.l1GtUnpack*process.hcalTimingMonitor*process.dqmEnv*process.dqmSaver)


