import FWCore.ParameterSet.Config as cms

process = cms.Process("test")
process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
process.source = cms.Source("EmptySource")

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

