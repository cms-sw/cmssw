import FWCore.ParameterSet.Config as cms
import CondTools.Ecal.db_credentials as auth

process = cms.Process("ProcessOne")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:EcalPedestals_tree.db'

process.MessageLogger = cms.Service("MessageLogger",
  debugModules = cms.untracked.vstring('*'),
  destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
  firstValue = cms.uint64(1),
  lastValue = cms.uint64(1),
  timetype = cms.string('timestamp'),
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
      tag = cms.string('EcalPedestals_laser_2016')
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
     OnlineDBSID = cms.string(db_service),
     OnlineDBUser = cms.string(db_user),
     OnlineDBPassword = cms.string( db_pwd ),
    Location = cms.string('***'),
    LocationSource = cms.string('Tree'),
    filename = cms.untracked.string('/afs/cern.ch/work/d/depasse/data/ana_ped_v3.root'),
    firstRun = cms.string('285550'),
    lastRun = cms.string('100000000'),
    debug = cms.bool(True),
    RunType = cms.untracked.int32(1)
  )
)

process.p = cms.Path(process.Test1)


