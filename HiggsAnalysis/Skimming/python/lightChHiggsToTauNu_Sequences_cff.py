import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.lightChHiggsToTauNu_HLTPaths_cfi import *
from HiggsAnalysis.Skimming.lightChHiggsToTauNu_Filter_cfi import *
lightChHiggsToTauNuHLTrigReport = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT")
)

lightChHiggsToTauNuSequence = cms.Sequence(lightChHiggsToTauNuHLTrigReport+lightChHiggsToTauNuHLTFilter+lightChHiggsToTauNuFilter)

