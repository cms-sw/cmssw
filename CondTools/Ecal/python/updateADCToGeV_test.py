import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(100000000000),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(100000000000),
    interval = cms.uint64(1)
)

process.load("CondCore.CondDB.CondDB_cfi")

process.CondDB.connect = 'sqlite_file:EcalADCToGeV.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDB, 
  logconnect = cms.untracked.string('sqlite_file:log.db'),   
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalADCToGeVConstantRcd'),
      tag = cms.string('EcalADCToGeV_test')
    )
  )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalADCToGeVAnalyzer",
  record = cms.string('EcalADCToGeVConstantRcd'),
  loggingOn= cms.untracked.bool(True),
  IsDestDbCheckedInQueryLog=cms.untracked.bool(True),
  SinceAppendMode=cms.bool(True),
  Source=cms.PSet(
    FileLowField = cms.string('ADCToGeV_Boff.xml'),
    FileHighField = cms.string('ADCToGeV_Bon.xml'),
    firstRun = cms.string('207149'),
    lastRun = cms.string('10000000'),
#     OnlineDBSID = cms.string('cms_omds_lb'),
    OnlineDBSID = cms.string('cms_orcon_adg'), # test on lxplus
    OnlineDBUser = cms.string('cms_ecal_r'),
    OnlineDBPassword = cms.string('3c4l_r34d3r'),
    LocationSource = cms.string('P5'),
    Location = cms.string('P5_Co'),
    GenTag = cms.string('GLOBAL'),
    RunType = cms.string('COSMICS')
  )                            
)

process.p = cms.Path(process.Test1)
