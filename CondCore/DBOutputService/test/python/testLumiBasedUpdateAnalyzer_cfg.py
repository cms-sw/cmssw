import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(100),
    timetype = cms.string('Lumi'),
    firstValue = cms.uint64(11),
    interval = cms.uint64(11)
)

process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    #timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:test_lumi.db'),
    preLoadConnectionString = cms.untracked.string('sqlite_file:test_lumi.db'),
    runNumber = cms.untracked.uint64(120000),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest'),
        timetype = cms.untracked.string('Lumi'),
        onlyAppendUpdatePolicy = cms.untracked.bool(True)
    ))
)

process.mytest = cms.EDAnalyzer("LumiBasedUpdateAnalyzer",
    record = cms.string('PedestalsRcd')
)

process.p = cms.Path(process.mytest)



