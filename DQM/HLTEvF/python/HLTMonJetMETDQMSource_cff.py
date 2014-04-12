import FWCore.ParameterSet.Config as cms

from DQM.HLTEvF.HLTMonJetMETDQMSource_cfi import *
#from DQM.HLTEvF.HLTMonMuonBits_cfi import *
#hltMonMuonDQM = cms.Path(hltMonMuDQM*hltMonMuBits)

hltmonjetmetTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/HLTEvF/data/HLTMonJetMET_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

#hltmonmuonseqClient = cms.Sequence(hltmonmuonqTester*hltmonmuonClient)

hltMonJMDQM = cms.Path(hltmonjetmetTester*hltMonjmDQM)

