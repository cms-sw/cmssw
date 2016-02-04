import FWCore.ParameterSet.Config as cms

l1tdemonTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TEmulator_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(500),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

l1tdemonseqClient = cms.Sequence(l1tdemonTester)

