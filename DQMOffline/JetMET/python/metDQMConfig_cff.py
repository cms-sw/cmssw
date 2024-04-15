import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

from DQMOffline.JetMET.metDQMConfig_cfi import *

#correction for type 1 done in JetMETDQMOfflineSource now
METDQMAnalyzerSequence = cms.Sequence(caloMetDQMAnalyzer*pfMetDQMAnalyzer*pfChMetDQMAnalyzer*pfMetT1DQMAnalyzer)

_METDQMAnalyzerSequenceWithPUPPI = cms.Sequence(caloMetDQMAnalyzer*pfMetDQMAnalyzer*pfChMetDQMAnalyzer*pfMetT1DQMAnalyzer*pfMetPUPPIDQMAnalyzer)

METDQMAnalyzerSequenceMiniAOD = cms.Sequence(pfMetDQMAnalyzerMiniAOD*pfPuppiMetDQMAnalyzerMiniAOD)

METDQMAnalyzerSequenceCosmics = cms.Sequence(caloMetDQMAnalyzer)

METDQMAnalyzerSequenceHI = cms.Sequence(caloMetDQMAnalyzer*pfMetDQMAnalyzer)

(~pp_on_AA).toReplaceWith(METDQMAnalyzerSequence, _METDQMAnalyzerSequenceWithPUPPI)
