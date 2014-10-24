import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:runinfo_pop_test.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'
process.CondDBCommon.DBParameters.messageLevel=cms.untracked.int32(3)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:logruninfo_pop_test.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('RunInfoRcd'),
        tag = cms.string('runinfo_test')
    ))
)

process.Test1 = cms.EDAnalyzer("RunInfoPopConAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('RunInfoRcd'),
    Source = cms.PSet(),
    loggingOn = cms.untracked.bool(True),
    IsDestDbCheckedInQueryLog = cms.untracked.bool(False)
)

process.p = cms.Path(process.Test1)



