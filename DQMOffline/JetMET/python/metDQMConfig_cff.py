import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.metDQMConfig_cfi import *

#correction for type 1 done in JetMETDQMOfflineSource now
METDQMAnalyzerSequence = cms.Sequence(caloMetDQMAnalyzer*pfMetDQMAnalyzer*pfChMetDQMAnalyzer*pfMetT1DQMAnalyzer)

METDQMAnalyzerSequenceMiniAOD = cms.Sequence(pfMetDQMAnalyzerMiniAOD*pfPuppiMetDQMAnalyzerMiniAOD)

METDQMAnalyzerSequenceCosmics = cms.Sequence(caloMetDQMAnalyzer)

METDQMAnalyzerSequenceHI = cms.Sequence(caloMetDQMAnalyzer*pfMetDQMAnalyzer)
# foo bar baz
# 9gzwJx8nAWeeG
# NOupZ9B0lHdD8
