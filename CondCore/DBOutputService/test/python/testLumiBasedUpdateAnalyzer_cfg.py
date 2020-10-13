import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(100),
    timetype = cms.string('Lumi'),
    firstValue = cms.uint64(11),
    interval = cms.uint64(11)
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(threshold = cms.untracked.string('DEBUG')),
    destinations = cms.untracked.vstring('cout')
)

process.OnlineDBOutputService = cms.Service("OnlineDBOutputService",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('.')
    ),
    #timetype = cms.untracked.string('runnumber'),
    jobName = cms.untracked.string("TestLumiBasedUpdate"),
    autoCommit = cms.untracked.bool(True),
    connect = cms.string('sqlite_file:test_lumi.db'),
    preLoadConnectionString = cms.untracked.string('sqlite_file:test_lumi.db'),
    #omsServiceUrl = cms.untracked.string('http://cmsoms-services.cms:9949/urn:xdaq-application:lid=100/getRunAndLumiSection'),
    lastLumiFile = cms.untracked.string('lastLumi.txt'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest'),
        timetype = cms.untracked.string('Lumi'),
        onlyAppendUpdatePolicy = cms.untracked.bool(True)
    ))
)

process.mytest = cms.EDAnalyzer("LumiBasedUpdateAnalyzer",
    record = cms.string('PedestalsRcd'),
    lastLumiFile = cms.untracked.string('lastLumi.txt')
    #omsServiceUrl = cms.untracked.string('http://cmsoms-services.cms:9949/urn:xdaq-application:lid=100/getRunAndLumiSection')
)

process.p = cms.Path(process.mytest)



