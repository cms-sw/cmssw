import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32( 260000 ),
                            firstLuminosityBlock = cms.untracked.uint32( 1 ),
                            numberEventsInRun = cms.untracked.uint32( 30 ),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(3),
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(30))

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    )
)

process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('.')
    ),
    jobName = cms.untracked.string("TestLumiBasedUpdate"),
    autoCommit = cms.untracked.bool(True),
    connect = cms.string('sqlite_file:test_lumi.db'),
    preLoadConnectionString = cms.untracked.string('sqlite_file:test_lumi.db'),
    lastLumiFile = cms.untracked.string('lastLumi.txt'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest'),
        timetype = cms.untracked.string('Lumi'),
        onlyAppendUpdatePolicy = cms.untracked.bool(True)
    ))
)

process.mytest = cms.EDAnalyzer("LumiBasedUpdateAnalyzer",
    record = cms.untracked.string('PedestalsRcd'),
    iovSize = cms.untracked.uint32(4),
    lastLumiFile = cms.untracked.string('lastLumi.txt')
)

process.p = cms.Path(process.mytest)



