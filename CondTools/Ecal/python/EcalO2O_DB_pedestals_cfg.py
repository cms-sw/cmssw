import FWCore.ParameterSet.Config as cms
import CondTools.Ecal.db_credentials as auth

process = cms.Process("ProcessOne")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:EcalPedestals_hlt.db'

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

db_reader_account = 'CMS_ECAL_R'
db_service,db_user,db_pwd = auth.get_db_credentials( db_reader_account )

#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#  process.CondDB,
#  timetype = cms.untracked.string('runnumber'),
#  toGet = cms.VPSet(
#    cms.PSet(
#      record = cms.string('EcalPedestalsRcd'),
#      tag = cms.string('EcalPedestals_hlt')
#    )
#  )
#)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDB,
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
#    GenTag = cms.string('GLOBAL'),
    GenTag = cms.string('LOCAL'),
    RunTag = cms.string('PEDESTAL'),
#    RunTag = cms.string('COSMIC'),
    firstRun = cms.string('240000'),
    lastRun = cms.string('100000000'),
     LocationSource = cms.string('P5'),
#    LocationSource = cms.string('File'),
#    LocationSource = cms.string('Tree'),
#    LocationSource = cms.string('Timestamp'),
#    LocationSource = cms.string('2017'),
        OnlineDBUser = cms.string(db_user),
        debug = cms.bool(False),
        OnlineDBPassword = cms.string(db_pwd),
        OnlineDBSID = cms.string(db_service),
    Location = cms.string('P5_Co'),
  )
)

process.p = cms.Path(process.Test1)


