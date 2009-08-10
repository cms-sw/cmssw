import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToTauTau_ElectronTau_HLTPaths_cfi import *
from HiggsAnalysis.Skimming.higgsToTauTau_ElectronTau_Filter_cfi import *

higgsToTauTauElectronTauHLTrigReport = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults::HLT")
)


higgsToTauTauElectronTauSequence = cms.Sequence(higgsToTauTauElectronTauHLTrigReport+higgsToTauTauElectronTauHLTFilter+higgsToTauTauElectronTauFilter)
