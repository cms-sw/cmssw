import FWCore.ParameterSet.Config as cms

# by default: the quality tests run at the end of each lumisection
DQMExample_qTester = cms.EDAnalyzer("QualityTester",
                       qtList = cms.untracked.FileInPath('DQMServices/Examples/test/DQMExample_QualityTest.xml'),
                       prescaleFactor = cms.untracked.int32(1),
                       #reportThreshold         = cms.untracked.string('black'),
                       getQualityTestsFromFile = cms.untracked.bool(True),
                       qtestOnEndJob           = cms.untracked.bool(True),
                       qtestOnEndRun           = cms.untracked.bool(False),
                       qtestOnEndLumi          = cms.untracked.bool(False),
                       testInEventloop         = cms.untracked.bool(False),
                       verboseQT               = cms.untracked.bool(True)
)
