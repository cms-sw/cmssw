# quality tests for L1 trigger occupancy
 
import FWCore.ParameterSet.Config as cms

l1TriggerOccupancyQualityTests = cms.EDAnalyzer("QualityTester",
    qtList=cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TriggerOccupancyQualityTests.xml'),
    QualityTestPrescaler=cms.untracked.int32(1),
    getQualityTestsFromFile=cms.untracked.bool(True),
    qtestOnEndLumi=cms.untracked.bool(True),
    verboseQT=cms.untracked.bool(True)
)