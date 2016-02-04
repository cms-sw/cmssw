import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.source = cms.Source("EventStreamHttpReader",
    sourceURL = cms.string('http://cmsmondev:50082/urn:xdaq-application:lid=29'),
    consumerPriority = cms.untracked.string('normal'),
    max_event_size = cms.int32(7000000),
    consumerName = cms.untracked.string('Test Consumer'),
    max_queue_depth = cms.int32(5),
    maxEventRequestRate = cms.untracked.double(10.0),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('*')
    ),
    headerRetryInterval = cms.untracked.int32(3)
)

process.dqmSource = cms.EDFilter("DQMSourceExample",
    monitorName = cms.untracked.string('YourSubsystemName')
)

process.qTester = cms.EDFilter("QualityTester",
    prescaleFactor = cms.untracked.int32(100000),
    qtList = cms.untracked.FileInPath('DQMServices/Examples/test/QualityTests.xml')
)

process.dqmClient = cms.EDFilter("DQMClientExample",
    prescaleLS = cms.untracked.int32(1),
    monitorName = cms.untracked.string('YourSubsystemName'),
    prescaleEvt = cms.untracked.int32(1000)
)

process.p = cms.Path(process.dqmSource+process.qTester+process.dqmClient+process.dqmEnv+process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'YourSubsystemName'

