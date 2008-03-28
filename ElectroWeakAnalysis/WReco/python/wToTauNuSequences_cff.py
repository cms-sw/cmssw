import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.WReco.wToTauNuHLTPaths_cfi import *
wToTauNuHLTSequence = cms.Sequence(wToTauNuHLTFilter)
wToTauNuAnalysis = cms.Sequence(wToTauNuHLTSequence)

