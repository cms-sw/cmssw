import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(10),
    timetype = cms.string('Run'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(True),
        authenticationPath = cms.untracked.string('.')
    ),
    timetype = cms.untracked.string('Run'),
    connect = cms.string('sqlite_file:anothertest.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest')
    ))
)

process.mytest = cms.EDAnalyzer("IOVPayloadEndOfJob",
    record = cms.string('PedestalsRcd')
)
process.p = cms.Path(process.mytest)



