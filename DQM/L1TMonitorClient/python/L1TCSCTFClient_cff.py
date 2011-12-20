import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TCSCTFClient_cfi import *
l1tcsctfTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TCSCTF_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(500),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

l1tcsctfseqClient = cms.Sequence(l1tcsctfTester*l1tcsctfClient)

