import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.FourVectorHLTOfflineClient_cfi import *

hltFourVectorQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQMOffline/Trigger/data/HLT_FourVector_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)


hltFourVectorSeqClient = cms.Sequence(hltFourVectorClient*hltFourVectorQTester)
