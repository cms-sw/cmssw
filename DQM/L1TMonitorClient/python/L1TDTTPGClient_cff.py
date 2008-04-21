import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TDTTPGClient_cfi import *
l1tdttpgqTester = cms.EDFilter("QualityTester",
    #  untracked string qtList = "DQM/L1TMonitorClient/test/dttpgQualityTests.xml"
    qtList = cms.untracked.string('l1tdttpgQualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(500),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

l1tdttpgseqClient = cms.Sequence(l1tdttpgqTester*l1tdttpgClient)

