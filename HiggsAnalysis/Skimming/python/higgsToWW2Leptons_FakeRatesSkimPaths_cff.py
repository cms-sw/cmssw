import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToWW2Leptons_FakeRatesSequences_cff import *
HWWFakeRatesFilterPath = cms.Path(higgsToWW2LeptonsFakeRatesSequence)

