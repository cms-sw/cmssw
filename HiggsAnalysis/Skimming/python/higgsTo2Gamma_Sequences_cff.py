import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsTo2Gamma_HLTPaths_cfi import *
from HiggsAnalysis.Skimming.higgsTo2Gamma_Filter_cfi import *
higgsTo2GammaHLTrigReport = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

higgsTo2GammaSequence = cms.Sequence(higgsTo2GammaHLTrigReport+higgsTo2GammaHLTFilter+higgsTo2GammaFilter)

