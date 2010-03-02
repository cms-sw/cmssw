import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToWW2Leptons_FakeRatesHLTPaths_cfi import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_FakeRatesFilter_cfi import *
higgsToWWFakeRatesTrigReport = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

higgsToWW2LeptonsFakeRatesSequence = cms.Sequence(higgsToWWFakeRatesTrigReport+higgsToWW2LeptonsFakeRatesHLTFilter+higgsToWW2LeptonsFakeRatesFilter)

