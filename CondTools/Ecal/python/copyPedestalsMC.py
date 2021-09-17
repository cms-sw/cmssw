import FWCore.ParameterSet.Config as cms
import CondTools.Ecal.db_credentials as auth

process = cms.Process("ProcessOne")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:EcalPedestals_mc2016_corrected.db'

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

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDB,
  logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
  timetype = cms.untracked.string('runnumber'),
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalPedestalsRcd'),
      tag = cms.string('EcalPedestals_MC')
    )
  )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalPedestalsAnalyzer",
  SinceAppendMode = cms.bool(True),
  record = cms.string('EcalPedestalsRcd'),
  loggingOn = cms.untracked.bool(True),
  Source = cms.PSet(
    GenTag = cms.string('***'),
    RunTag = cms.string('***'),
        OnlineDBUser = cms.string(db_user),
        OnlineDBPassword = cms.string(db_pwd),
        OnlineDBSID = cms.string(db_service),
    Location = cms.string('***'),
    LocationSource = cms.string('MC'),
    filename = cms.untracked.string('./Pedestals_278181.xml'),
#    filename = cms.untracked.string('./Pedestals_303848.xml'),
#    filename = cms.untracked.string('./Pedestals_304211.xml'),
    firstRun = cms.string('1'),
    lastRun = cms.string('100000000'),
    corrected = cms.untracked.bool(True),
    debug = cms.bool(False)
  )
)

process.p = cms.Path(process.Test1)


