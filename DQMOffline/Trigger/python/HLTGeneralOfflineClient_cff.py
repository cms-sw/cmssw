import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTGeneralOfflineClient_cfi import *

hltGeneralQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQMOffline/Trigger/data/HLT_General_QualityTests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)


hltGeneralSeqClient = cms.Sequence(hltGeneralClient*hltGeneralQTester)
