import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string("sqlite_file:testLogging.db")

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(10),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(2)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest')
    ))
)

process.mytest = cms.EDAnalyzer("MyDataAnalyzer",
    record = cms.string('PedestalsRcd'),
    loggingOn = cms.untracked.bool(True)
)

process.p = cms.Path(process.mytest)



