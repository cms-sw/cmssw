import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
# process.CondDBCommon.connect = 'sqlite_file:DB.db'



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
                                firstValue = cms.uint64(1),
                                lastValue = cms.uint64(1),
                                timetype = cms.string('runnumber'),
                                interval = cms.uint64(1)
                            )

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals_v5_online')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalPedestalsRcd'),
        tag = cms.string('EcalPedestals_v5_online')
    ))
)

#    logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_21X_POPCONLOG'),



process.Test1 = cms.EDAnalyzer("ExTestEcalPedestalsAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('EcalPedestalsRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        GenTag = cms.string('LOCAL'),
        firstRun = cms.string('42058'),
        lastRun = cms.string('100000000'),
        LocationSource = cms.string('P5'),
        OnlineDBUser = cms.string('CHANGE_HERE'),
        debug = cms.bool(True),
        Location = cms.string('P5_Co'),
        OnlineDBPassword = cms.string('CHANGE_HERE'),
        OnlineDBSID = cms.string('CHANGE_HERE')
    )
)

process.p = cms.Path(process.Test1)


