import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_34X_ECAL'
#process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDBCommon.connect = 'sqlite_file:EcalSRSettings_v00_beam10.db'


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
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('EcalSRSettingsRcd'),
            tag = cms.string('EcalSRSettings_beam2010_v01_offline')
        )
    )
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('EcalSRSettingsRcd'),
            tag = cms.string('EcalSRSettings_beam2010_v01_offline')
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
        OnlineDBUser = cms.string('cms_ecal_r'),
        debug = cms.bool(True),
        OnlineDBPassword = cms.string('xxxxx'),
        OnlineDBSID = cms.string('cms_orcon_prod'),
        location = cms.string('P5_Co'),
        runtype = cms.string('Physics'), 
        gentag = cms.string('global'),
    )
)

process.p = cms.Path(process.Test1)


