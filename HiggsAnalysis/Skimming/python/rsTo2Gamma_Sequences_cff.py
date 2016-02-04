import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.rsTo2Gamma_HLTPaths_cfi import *
from HiggsAnalysis.Skimming.rsTo2Gamma_Filter_cfi import *
rsTo2GammaHLTrigReport = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

rsTo2GammaSequence = cms.Sequence(rsTo2GammaHLTrigReport+rsTo2GammaHLTFilter+rsTo2GammaFilter)

