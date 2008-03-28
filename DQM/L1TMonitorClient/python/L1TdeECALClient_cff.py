import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1TdeECALClient_cfi import *
l1tdeEcalqTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.string('deECALQualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(500),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

l1tdeEcalseqClient = cms.Sequence(l1tdeEcalqTester*l1tdeEcalClient)

