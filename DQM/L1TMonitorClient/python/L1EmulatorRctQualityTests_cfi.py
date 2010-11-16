# quality tests for L1 RCT trigger
 
import FWCore.ParameterSet.Config as cms

l1EmulatorRctQualityTests = cms.EDAnalyzer("QualityTester",
    qtList=cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1EmulatorRctQualityTests.xml'),
    QualityTestPrescaler=cms.untracked.int32(1),
    getQualityTestsFromFile=cms.untracked.bool(True),
    qtestOnEndLumi=cms.untracked.bool(True),
    verboseQT=cms.untracked.bool(True)
)

