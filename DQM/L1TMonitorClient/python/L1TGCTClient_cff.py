import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TGCTClient_cfi import *
l1tGctqTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TGCT_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

l1tGctseqClient = cms.Sequence(l1tGctqTester*l1tGctClient)

