import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToTauTau_DoubleTauHLTPaths_cfi import *
zToTauTau_DoubleTauHLTSequence = cms.Sequence(zToTauTau_DoubleTauHLTFilter)
zToTauTau_DoubleTauAnalysis = cms.Sequence(zToTauTau_DoubleTauHLTSequence)

