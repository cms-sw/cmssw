# quality tests for HCAL TPG trigger
 
import FWCore.ParameterSet.Config as cms

l1EmulatorHcalQualityTests = cms.EDAnalyzer("QualityTester",
    qtList=cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1EmulatorHcalQualityTests.xml'),
    QualityTestPrescaler=cms.untracked.int32(1),
    getQualityTestsFromFile=cms.untracked.bool(True),
    qtestOnEndLumi=cms.untracked.bool(True),
    verboseQT=cms.untracked.bool(True)
)

