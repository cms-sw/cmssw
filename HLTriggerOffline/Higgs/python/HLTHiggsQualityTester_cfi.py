import FWCore.ParameterSet.Config as cms

hltHiggsQualityTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath(
        'HLTriggerOffline/Higgs/data/HLTHiggsQualityTest.xml'
    ),
    #reportThreshold         = cms.untracked.string('black'),
    prescaleFactor          = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndJob           = cms.untracked.bool(True),
    qtestOnEndLumi          = cms.untracked.bool(False),
    testInEventloop         = cms.untracked.bool(False),
    verboseQT               = cms.untracked.bool(False)
)
