import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:pop_test2.db'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    withWrapper = cms.untracked.bool(True),
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('ThisJob'),
        tag = cms.string('Example_tag1')
    )
        )
)

process.Test1 = cms.EDAnalyzer("ExPopConEfficiency",
    record = cms.string('ThisJob'),
    Source = cms.PSet(
        params = cms.untracked.vdouble(0.1, 0.95, 1.0, 5.5),
        since = cms.untracked.int64(701),
        type = cms.untracked.string('Pt')
    ),
    targetDBConnectionString = cms.untracked.string('sqlite_file:pop_test.db'),
    loggingOn = cms.untracked.bool(True),
    IsDestDbCheckedInQueryLog = cms.untracked.bool(True)
)

process.Test2 = cms.EDAnalyzer("ExPopConEfficiency",
    record = cms.string('ThisJob'),
    Source = cms.PSet(
        params = cms.untracked.vdouble(0.85, 0.0, 0.9, 2.3),
        since = cms.untracked.int64(930),
        type = cms.untracked.string('Eta')
    ),
    targetDBConnectionString = cms.untracked.string('sqlite_file:pop_test.db'),
    loggingOn = cms.untracked.bool(True),
    IsDestDbCheckedInQueryLog = cms.untracked.bool(True)
)

process.Test3 = cms.EDAnalyzer("ExPopConEfficiency",
    record = cms.string('ThisJob'),
    Source = cms.PSet(
        params = cms.untracked.vdouble(0.92, 0.0, 0.8, 2.5),
        since = cms.untracked.int64(1240),
        type = cms.untracked.string('Eta')
    ),
    targetDBConnectionString = cms.untracked.string('sqlite_file:pop_test.db'),
    loggingOn = cms.untracked.bool(True),
    IsDestDbCheckedInQueryLog = cms.untracked.bool(True)
)

process.Test4 = cms.EDAnalyzer("ExPopConEfficiency",
    record = cms.string('ThisJob'),
    Source = cms.PSet(
        params = cms.untracked.vdouble(0.1, 0.95, 1.0, 9.5),
        since = cms.untracked.int64(1511),
        type = cms.untracked.string('Pt')
    ),
    targetDBConnectionString = cms.untracked.string('sqlite_file:pop_test.db'),
    loggingOn = cms.untracked.bool(True),
    IsDestDbCheckedInQueryLog = cms.untracked.bool(True)
)

process.p = cms.Path(process.Test1 +
                      process.Test2 +
                      process.Test3 +
                      process.Test4
                      )


# process.p = cms.Path(process.TestN) 















