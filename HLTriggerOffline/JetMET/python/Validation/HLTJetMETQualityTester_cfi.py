import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMQualityTester import DQMQualityTester
hltJetMetQualityTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath(
        'HLTriggerOffline/JetMET/data/QualityTests.xml'
    ),
    #reportThreshold         = cms.untracked.string('black'),
    prescaleFactor          = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndJob           = cms.untracked.bool(False),
    qtestOnEndRun           = cms.untracked.bool(True),
    qtestOnEndLumi          = cms.untracked.bool(False),
    testInEventloop         = cms.untracked.bool(False),
    verboseQT               = cms.untracked.bool(False)
)
