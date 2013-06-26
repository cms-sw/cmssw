import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:test.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string('t0_test')
    ))
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(2),
    interval   = cms.uint64(1)
)

process.t0 = cms.EDAnalyzer("DTT0PopConAnalyzer",
    name = cms.untracked.string('DTT0'),
    record = cms.string('DTT0Rcd'),
    Source = cms.PSet(
        tag = cms.string('t0_test'),
        file = cms.string('t0.txt'),
        run = cms.uint32(1)
    ),
    SinceAppendMode = cms.bool(True)
)

process.p = cms.Path(process.t0)

