import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("EmptyIOVSource",
    lastValue= cms.uint64(4532789567110001000),
    timetype = cms.string('Time'),
    firstValue = cms.uint64(4532789567110000000),
    interval = cms.uint64(20)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    timetype = cms.untracked.string('Time'),
    connect = cms.string('sqlite_file:test.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest')
    ))
)

process.mytest = cms.EDAnalyzer("Timestamp",
    record = cms.string('PedestalsRcd')
)

process.p = cms.Path(process.mytest)



