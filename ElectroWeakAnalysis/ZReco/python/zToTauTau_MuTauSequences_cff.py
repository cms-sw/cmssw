import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToTauTau_MuTauHLTPaths_cfi import *
zToTauTau_MuTauHLTSequence = cms.Sequence(zToTauTau_MuTauHLTFilter)
zToTauTau_MuTauAnalysis = cms.Sequence(zToTauTau_MuTauHLTSequence)

