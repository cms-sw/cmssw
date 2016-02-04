import FWCore.ParameterSet.Config as cms

##from DQM.L1TMonitorClient.L1TRCTClient_cfi import *
l1tRctqTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TRCT_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

l1tRctseqClient = cms.Sequence(l1tRctqTester)

