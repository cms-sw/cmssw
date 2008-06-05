import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20000)
)
source = cms.Source("EmptySource")

dqmSource = cms.EDFilter("DQMSourceExample",
    monitorName = cms.untracked.string('TestSystem'),
    prescaleEvt = cms.untracked.int32(10)
)

qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQMServices/Examples/test/QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(2)
)

dqmClient = cms.EDFilter("DQMClientExample",
    prescaleLS = cms.untracked.int32(-1),
    monitorName = cms.untracked.string('TestSystem'),
    prescaleEvt = cms.untracked.int32(1000)
)

p = cms.Path(dqmSource+qTester+dqmClient)

