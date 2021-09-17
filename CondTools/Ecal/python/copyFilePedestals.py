import FWCore.ParameterSet.Config as cms
import CondTools.Ecal.db_credentials as auth

process = cms.Process("ProcessOne")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:EcalPedestals_mc.db'

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
#      tag = cms.string('EcalPedestals_-0.1_-0.1')
#      tag = cms.string('EcalPedestals_-0.3_-0.5')
#      tag = cms.string('EcalPedestals_-0.5_-1')
#      tag = cms.string('EcalPedestals_-1.5_-5')
#      tag = cms.string('EcalPedestals_-1_-3')
#      tag = cms.string('EcalPedestals_-2_-10')
#      tag = cms.string('EcalPedestals_0.1_0.1')
#      tag = cms.string('EcalPedestals_0.3_0.5')
#      tag = cms.string('EcalPedestals_0.5_1')
#      tag = cms.string('EcalPedestals_1.5_5')
#      tag = cms.string('EcalPedestals_1_3')
#      tag = cms.string('EcalPedestals_2_10')
      tag = cms.string('EcalPedestals_0_0')
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
    LocationSource = cms.string('File'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_-0.1_-0.1.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_-0.3_-0.5.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_-0.5_-1.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_-1.5_-5.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_-1_-3.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_-2_-10.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_0.1_0.1.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_0.3_0.5.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_0.5_1.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_1.5_5.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_1_3.dat'),
#    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_2_10.dat'),
    filename = cms.untracked.string('/afs/cern.ch/user/s/shervin/public/forJean/6a9a2818932fce79d8222768ba4f2ad3f60f894c_0_0.dat'),
    firstRun = cms.string('1'),
    lastRun = cms.string('100000000'),
    debug = cms.bool(True)
  )
)

process.p = cms.Path(process.Test1)


