# quality tests for L1 trigger occupancy
 
import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMQualityTester import DQMQualityTester
l1TriggerOccupancyQualityTests = DQMQualityTester(
    qtList=cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TriggerOccupancyQualityTests.xml'),
    QualityTestPrescaler=cms.untracked.int32(1),
    getQualityTestsFromFile=cms.untracked.bool(True),
    testInEventloop=cms.untracked.bool(False),
    qtestOnEndLumi=cms.untracked.bool(True),
    qtestOnEndRun=cms.untracked.bool(True),
    qtestOnEndJob=cms.untracked.bool(False),
    reportThreshold=cms.untracked.string(""),
    verboseQT=cms.untracked.bool(True)
)