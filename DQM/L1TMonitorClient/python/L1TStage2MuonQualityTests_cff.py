# quality tests for stage 2 L1T muons 

import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMQualityTester import DQMQualityTester
l1TStage2MuonQualityTests = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TStage2MuonQualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    testInEventloop = cms.untracked.bool(False),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndRun = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(False),
    reportThreshold = cms.untracked.string(""),
    verboseQT = cms.untracked.bool(True)
)

l1TStage2MuonQualityTestsCollisions = l1TStage2MuonQualityTests.clone()
l1TStage2MuonQualityTestsCollisions.qtList = cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TStage2MuonQualityTestsCollisions.xml')
