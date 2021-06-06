import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:DB.db'

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")
process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")

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
        record = cms.string('EcalChannelStatusRcd'),
        tag = cms.string('EcalChannelStatus_online')
    ))
)

process.es_prefer_ChannelStatusDB = cms.ESPrefer("PoolDBESSource", "")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalChannelStatusRcd'),
        tag = cms.string('EcalChannelStatus_online')
    ))
)

process.Test1 = cms.EDAnalyzer("ExTestEcalChannelStatusAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('EcalChannelStatusRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        GenTag = cms.string('GLOBAL'),           # LOCAL or GLOBAL
        RunType = cms.string('COSMIC'),          # PEDESTAL, LASER or COSMIC
        firstRun = cms.string('112639'),   
        lastRun = cms.string('112639'),
        LocationSource = cms.string('P5'),
        OnlineDBUser = cms.string('cms_ecal_r'),
        debug = cms.bool(True),
        Location = cms.string('P5_Co'),
        OnlineDBPassword = cms.string('******'),
        OnlineDBSID = cms.string('cms_testbeam'),
        )
)

process.p = cms.Path(process.Test1)


