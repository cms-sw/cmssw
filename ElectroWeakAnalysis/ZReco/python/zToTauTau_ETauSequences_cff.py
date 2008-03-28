import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToTauTau_ETauHLTPaths_cfi import *
zToTauTauETauHLTSequence = cms.Sequence(zToTauTauETauHLTFilter)
zToTauTau_ETauAnalysis = cms.Sequence(zToTauTauETauHLTSequence)

