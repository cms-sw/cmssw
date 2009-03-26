import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:testhv.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:log.db'),
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
#        onlineDB = cms.string('oracle://cms_orcoff_int/CMS_COND_20X_DT'),
        onlineDB = cms.string('oracle://cms_omds_lb/cms_dt_hv_pvss_cond'),
#        onlineDB = cms.string('sqlite_file:dummy_online.db'),
        bufferDB = cms.string('sqlite_file:buffer.db'),
        tag = cms.string('hv_test'),
        onlineAuthentication = cms.string('.'),
        sinceYear  = cms.int32(2008),
        sinceMonth = cms.int32(  10),
        sinceDay   = cms.int32(   1),
        untilYear  = cms.int32(2008),
        untilMonth = cms.int32(  10),
        untilDay   = cms.int32(  25),
        mapVersion = cms.string('hvChannels_090326')
    ),
    SinceAppendMode = cms.bool(True),
    record = cms.string('DTHVStatusRcd'),
    loggingOn = cms.untracked.bool(True),
    debug = cms.bool(False)
)

process.p = cms.Path(process.hv_o2o)

