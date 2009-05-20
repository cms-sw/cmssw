import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_HLTPaths_cfi import *
from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_Filter_cfi import *

higgsToTauTauLeptonTauHLTrigReport = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults::HLT")
)


higgsToTauTauLeptonTauSequence = cms.Sequence(higgsToTauTauLeptonTauHLTrigReport+higgsToTauTauLeptonTauHLTFilter+higgsToTauTauLeptonTauFilter)
