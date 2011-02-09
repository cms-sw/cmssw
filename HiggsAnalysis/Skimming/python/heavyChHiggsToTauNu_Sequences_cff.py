import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.heavyChHiggsToTauNu_HLTPaths_cfi import *
from HiggsAnalysis.Skimming.heavyChHiggsToTauNu_Filter_cfi import *
heavyChHiggsToTauNuHLTrigReport = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

heavyChHiggsToTauNuSequence = cms.Sequence(heavyChHiggsToTauNuHLTrigReport+heavyChHiggsToTauNuHLTFilter+heavyChHiggsToTauNuFilter)

