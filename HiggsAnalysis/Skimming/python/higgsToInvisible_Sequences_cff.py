import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToInvisible_HLTPaths_cfi import *
higgsToInvisibleTrigReport = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

higgsToInvisibleSequence = cms.Sequence(higgsToInvisibleTrigReport+higgsToInvisibleHLTFilter)

