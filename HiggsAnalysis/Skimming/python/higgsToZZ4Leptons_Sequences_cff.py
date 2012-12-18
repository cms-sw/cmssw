import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToZZ4Leptons_HLTPaths_cfi import *
from HiggsAnalysis.Skimming.higgsToZZ4Leptons_Filter_cfi import *
higgsToZZ4HLTrigReport = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

higgsToZZ4LeptonsSequence = cms.Sequence(higgsToZZ4HLTrigReport+higgsToZZ4LeptonsHLTFilter+higgsToZZ4LeptonsFilter)

