import FWCore.ParameterSet.Config as cms

from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_HLTPaths_cfi import *
higgsToTauTauLeptonTauSequence = cms.Sequence(higgsToTauTauLeptonTauHLTFilter)

