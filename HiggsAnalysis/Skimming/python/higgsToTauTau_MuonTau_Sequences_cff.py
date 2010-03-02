import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_HLTPaths_cfi import *
from HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_Filter_cfi import *

higgsToTauTauMuonTauHLTrigReport = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults::HLT")
)


higgsToTauTauMuonTauSequence = cms.Sequence(higgsToTauTauMuonTauHLTrigReport+higgsToTauTauMuonTauHLTFilter+higgsToTauTauMuonTauFilter)
