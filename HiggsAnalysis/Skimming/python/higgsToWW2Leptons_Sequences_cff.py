import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToWW2Leptons_HLTPaths_cfi import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_Filter_cfi import *
higgsToWWTrigReport = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

higgsToWW2LeptonsSequence = cms.Sequence(higgsToWWTrigReport+higgsToWW2LeptonsHLTFilter+higgsToWW2LeptonsFilter)

