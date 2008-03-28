import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToTauTau_EMuHLTPaths_cfi import *
zToTauTau_EMuHLTSequence = cms.Sequence(zToTauTau_EMuHLTFilter)
zToTauTau_EMuAnalysis = cms.Sequence(zToTauTau_EMuHLTSequence)

