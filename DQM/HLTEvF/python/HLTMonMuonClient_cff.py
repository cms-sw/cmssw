import FWCore.ParameterSet.Config as cms

from DQM.HLTEvF.HLTMonMuonClient_cfi import *

hltmonmuonqTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/HLTEvF/data/HLTMonMuon_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

hltmonmuonseqClient = cms.Sequence(hltmonmuonqTester*hltmonmuonClient)

