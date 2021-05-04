import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMQualityTester import DQMQualityTester
hltQTExample = DQMQualityTester(
    qtList = cms.untracked.FileInPath('HLTriggerOffline/Common/data/HltQTExample.xml'),
    #QualityTestPrescaler = cms.untracked.int32(1)
    reportThreshold = cms.untracked.string('black'),
    prescaleFactor = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndJob=cms.untracked.bool(True),
    testInEventloop=cms.untracked.bool(False),
    qtestOnEndLumi=cms.untracked.bool(False)
)
