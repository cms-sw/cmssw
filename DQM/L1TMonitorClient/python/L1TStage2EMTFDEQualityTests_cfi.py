# quality tests for L1T Stage2 EMTF Data vs Emulator 
 
import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMQualityTester import DQMQualityTester
l1TStage2EMTFDEQualityTests = DQMQualityTester(
    qtList=cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1TStage2EMTFDEQualityTests.xml'),
    QualityTestPrescaler=cms.untracked.int32(1),
    getQualityTestsFromFile=cms.untracked.bool(True),
    testInEventloop=cms.untracked.bool(False),
    qtestOnEndLumi=cms.untracked.bool(True),
    qtestOnEndRun=cms.untracked.bool(True),
    qtestOnEndJob=cms.untracked.bool(False),
    reportThreshold=cms.untracked.string(""),
    verboseQT=cms.untracked.bool(True)
)
