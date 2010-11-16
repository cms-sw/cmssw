# quality tests for L1 GtExternal trigger objects
 
import FWCore.ParameterSet.Config as cms

l1EmulatorObjGtExternalQualityTests = cms.EDAnalyzer("QualityTester",
    qtList=cms.untracked.FileInPath('DQM/L1TMonitorClient/data/L1EmulatorObjGtExternalQualityTests.xml'),
    QualityTestPrescaler=cms.untracked.int32(1),
    getQualityTestsFromFile=cms.untracked.bool(True),
    qtestOnEndLumi=cms.untracked.bool(True),
    verboseQT=cms.untracked.bool(True)
)

