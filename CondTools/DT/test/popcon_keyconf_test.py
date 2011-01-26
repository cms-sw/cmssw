import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:testconf.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(
    cms.PSet(
        record = cms.string('DTCCBConfigRcd'),
        tag = cms.string('conf_test'),
        timetype = cms.untracked.string('runnumber')
    ),
    cms.PSet(
        record = cms.string('keyedConfBricks'),
        tag = cms.string('keyedConfBricks_V01'),
        timetype = cms.untracked.string('hash'),
        withWrapper = cms.untracked.bool(True),
        outOfOrder = cms.untracked.bool(True)
    ),
    cms.PSet(
        record = cms.string('keyedConfListIOV'),
        tag = cms.string('keyedConfListIOV_V01'),
        timetype = cms.untracked.string('runnumber'),
        withWrapper = cms.untracked.bool(True),
        outOfOrder = cms.untracked.bool(False)
    )
    )
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval   = cms.uint64(1)
)

process.essource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    DumpStat=cms.untracked.bool(True),                           
    toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('DTKeyedConfigListRcd'),
    tag = cms.string('keyedConfListIOV_V01')
    ),
    cms.PSet(
    record = cms.string('DTKeyedConfigContainerRcd'),
    tag = cms.string('keyedConfBricks_V01')
    )
    )
)

process.conf_o2o = cms.EDAnalyzer("DTKeyedConfigPopConAnalyzer",
    name = cms.untracked.string('DTCCBConfig'),
    Source = cms.PSet(
        DBParameters = cms.PSet(
            messageLevel = cms.untracked.int32(0),
            authenticationPath = cms.untracked.string('.')
        ),
        onlineDB = cms.string('sqlite_file:dummy_online.db'),
        minBrick = cms.untracked.int32(0),
        maxBrick = cms.untracked.int32(40),
        minRun = cms.untracked.int32(99),
        maxRun = cms.untracked.int32(50000),
        tag = cms.string('conf_test'),
        container = cms.string('keyedConfBricks'),
        onlineAuthentication = cms.string('.')
    ),
    SinceAppendMode = cms.bool(True),
    record = cms.string('DTCCBConfigRcd'),
    loggingOn = cms.untracked.bool(True),
    debug = cms.bool(False)
)

process.p = cms.Path(process.conf_o2o)

