import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_34X_ECAL'
process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDBCommon.connect = 'sqlite_file:EcalDAQTowerStatus.db'



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
            record = cms.string('EcalDAQTowerStatusRcd'),
            tag = cms.string('EcalDAQTowerStatus_online')
        )
    )
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    #logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG'),
    logconnect = cms.untracked.string('sqlite_file:/tmp/EcalDAQDBlog.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('EcalDAQTowerStatusRcd'),
            tag = cms.string('EcalDAQTowerStatus_online')
        )
    )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalDAQAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('EcalDAQTowerStatusRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        firstRun = cms.string('121756'),
        lastRun = cms.string('100000000'),
        OnlineDBUser = cms.string('cms_ecal_r'),
        OnlineDBPassword = cms.string('******'),
        OnlineDBSID = cms.string('cms_omds_lb'),
        location = cms.string('P5_Co'),
#        runtype = cms.string('Cosmic'), 
#        runtype = cms.string('Cosmic'), 
        runtype = cms.string('PHYSICS'), 
        gentag = cms.string('GLOBAL'),
        debug = cms.bool(False)
    )
)

process.p = cms.Path(process.Test1)


