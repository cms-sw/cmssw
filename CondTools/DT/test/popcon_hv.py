import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:testhv.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'

#process.DTHVCheckByAbsoluteValues = cms.Service("DTHVCheckByAbsoluteValues")
process.DTHVCheckWithHysteresis = cms.Service("DTHVCheckWithHysteresis")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    timetype = cms.untracked.string('timestamp'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTHVStatusRcd'),
        tag = cms.string('hv_test')
    ))
)

process.source = cms.Source("EmptyIOVSource",
#    timetype = cms.string('runnumber'),
    timetype = cms.string('timestamp'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(5),
    interval   = cms.uint64(1)
)

process.hv_o2o = cms.EDAnalyzer("DTHVStatusPopConAnalyzer",
    name = cms.untracked.string('DTHVStatus'),
    Source = cms.PSet(
        DBParameters = cms.PSet(
            messageLevel = cms.untracked.int32(0),
            authenticationPath = cms.untracked.string('.')
        ),
        onlineDB = cms.string('sqlite_file:dummy_online.db'),
        utilDB   = cms.string('sqlite_file:dummy_online.db'),
        bufferDB = cms.string('sqlite_file:bufferHV.db'),
        tag = cms.string('hv_test'),
        onlineAuthentication = cms.string('.'),
        sinceYear   = cms.int32(2009),
        sinceMonth  = cms.int32(  12),
        sinceDay    = cms.int32(   6),
        sinceHour   = cms.int32(   7),
        sinceMinute = cms.int32(   0),
        sinceSecond = cms.int32(   1),
        untilYear   = cms.int32(2009),
        untilMonth  = cms.int32(  12),
        untilDay    = cms.int32(   6),
        untilHour   = cms.int32(  17),
        untilMinute = cms.int32(   1),
        untilSecond = cms.int32(   0),
        dumpAtStart = cms.bool(True),
        dumpAtEnd   = cms.bool(False),
        bwdTime      = cms.int64(3600),
        fwdTime      = cms.int64(3600),
        minTime      = cms.int64(   1),
        mapVersion = cms.string('hvChannels_090326'),
        splitVersion = cms.string('hvChanSplit_090818')
    ),
    SinceAppendMode = cms.bool(True),
    record = cms.string('DTHVStatusRcd'),
    loggingOn = cms.untracked.bool(True),
    debug = cms.bool(False)
)

process.p = cms.Path(process.hv_o2o)

