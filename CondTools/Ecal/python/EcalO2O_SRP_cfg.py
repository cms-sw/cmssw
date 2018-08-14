import FWCore.ParameterSet.Config as cms
import CondTools.Ecal.db_credentials as auth

process = cms.Process("ProcessOne")

process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDB.connect = 'oracle://cms_orcon_prod/CMS_COND_34X_ECAL'
#process.CondDB.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDB.connect = 'sqlite_file:EcalSRSettings.db'


process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1000000),
    lastValue = cms.uint64(1000000),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('EcalSRSettingsRcd'),
            tag = cms.string('EcalSRSettings_v01_offline')
        )
    )
)

db_service,db_user,db_pwd = auth.get_readOnly_db_credentials()

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('EcalSRSettingsRcd'),
            tag = cms.string('EcalSRSettings_v01_offline')
        )
    )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalSRPAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('EcalSRSettingsRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        firstRun = cms.string('160970'),
        lastRun = cms.string('100000000'),
        debug = cms.bool(True),
     OnlineDBSID = cms.string(db_service),
     OnlineDBUser = cms.string(db_user),
     OnlineDBPassword = cms.string( db_pwd ),
        location = cms.string('P5_Co'),
        runtype = cms.string('Physics'), 
        gentag = cms.string('global'),
    )
)

process.p = cms.Path(process.Test1)


