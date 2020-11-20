import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.load("CondCore.DBCommon.CondDBCommon_cfi")
# process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
process.CondDBCommon.connect = 'sqlite_file:DB.db'



process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring('*')
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
            tag = cms.string('EcalDAQTowerStatus_mc')
        )
    )
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('EcalDAQTowerStatusRcd'),
            tag = cms.string('EcalDAQTowerStatus_mc')
        )
    )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalDAQAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('EcalDAQTowerStatusRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        firstRun = cms.string('123000'),
        lastRun = cms.string('100000000'),
        OnlineDBUser = cms.string('cms_ecal_r'),
        debug = cms.bool(True),
        OnlineDBPassword = cms.string('*****'),
        OnlineDBSID = cms.string('cms_omds_lb'),
        location = cms.string('P5_Co'),
#        runtype = cms.string('Cosmic'), 
#        runtype = cms.string('Cosmic'), 
        runtype = cms.string('Physics'), 
        gentag = cms.string('global')
    )
)

process.p = cms.Path(process.Test1)


