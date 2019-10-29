import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMQualityTester import DQMQualityTester
hltqTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/HLTEvF/data/hltBitsQualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)


hltqtest = cms.Sequence(hltqTester)
