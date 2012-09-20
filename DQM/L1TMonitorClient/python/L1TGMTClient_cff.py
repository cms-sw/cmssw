import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TGMTClient_cfi import *
l1tgmtqTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TGMT_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)


l1tgmtseqClient = cms.Sequence(l1tgmtqTester*l1tgmtClient)
