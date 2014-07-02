import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.metDQMConfig_cfi import *

#METDQMAnalyzerSequence = cms.Sequence(caloMetDQMAnalyzer*tcMetDQMAnalyzer*pfMetDQMAnalyzer)
METDQMAnalyzerSequence = cms.Sequence(caloMetDQMAnalyzer*pfMetDQMAnalyzer)

METDQMAnalyzerSequenceCosmics = cms.Sequence(caloMetDQMAnalyzer)

METDQMAnalyzerSequenceHI = cms.Sequence(caloMetDQMAnalyzer*pfMetDQMAnalyzer)
