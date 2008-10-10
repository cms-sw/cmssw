import FWCore.ParameterSet.Config as cms

from DQM.DTMonitorClient.dtChamberEfficiencyTest_cfi import *
from DQM.DTMonitorClient.dtSegmentAnalysisTest_cfi import *
segmentTest.detailedAnalysis = True

dtqTester = cms.EDFilter("QualityTester",
                         #reportThreshold = cms.untracked.string('red'),
                         prescaleFactor = cms.untracked.int32(1),
                         qtList = cms.untracked.FileInPath('DQM/DTMonitorClient/test/QualityTests.xml'),
                         getQualityTestsFromFile = cms.untracked.bool(True)
                         )


dtClients = cms.Sequence(chamberEfficiencyTest*segmentTest*dtqTester)
