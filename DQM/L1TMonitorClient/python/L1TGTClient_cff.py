import FWCore.ParameterSet.Config as cms

l1tgtqTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TGT_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)


l1tgtseqClient = cms.Sequence(l1tgtqTester)
