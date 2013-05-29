import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TRPCTFClient_cfi import *
l1trpctfqTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TRPCTF_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

l1trpctfseqClient = cms.Sequence(l1trpctfqTester*l1trpctfClient)

