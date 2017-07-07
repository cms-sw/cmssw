import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
#process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
process.CondDBCommon.connect = 'sqlite_file:DB.db'

process.MessageLogger = cms.Service("MessageLogger",
  debugModules = cms.untracked.vstring('*'),
  destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
  firstValue = cms.uint64(1),
  lastValue = cms.uint64(1),
  timetype = cms.string('runnumber'),
  interval = cms.uint64(1)
)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
  process.CondDBCommon,
  timetype = cms.untracked.string('runnumber'),
  toGet = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalPedestalsRcd'),
      tag = cms.string('EcalPedestals_hlt')
    )
  )
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDBCommon,
  logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
  timetype = cms.untracked.string('runnumber'),
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalPedestalsRcd'),
      tag = cms.string('EcalPedestals_hlt')
    )
  )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalPedestalsAnalyzer",
  SinceAppendMode = cms.bool(True),
  record = cms.string('EcalPedestalsRcd'),
  loggingOn = cms.untracked.bool(True),
  Source = cms.PSet(
    GenTag = cms.string('GLOBAL'),
    RunTag = cms.string('PEDESTAL'),
#        RunTag = cms.string('COSMIC'),
    firstRun = cms.string('238500'),
    lastRun = cms.string('100000000'),
    LocationSource = cms.string('P5'),
    OnlineDBUser = cms.string('cms_ecal_r'),
    OnlineDBPassword = cms.string('3c4l_r34d3r'),
#        OnlineDBUser = cms.string('cms_ecal_conf'),
#        OnlineDBPassword = cms.string('0r4cms_3c4lc0nf'),
    debug = cms.bool(True),
    Location = cms.string('P5_Co'),
#        OnlineDBSID = cms.string('INT2R_LB')
    OnlineDBSID = cms.string('cms_omds_adg')
  )
)

process.p = cms.Path(process.Test1)


