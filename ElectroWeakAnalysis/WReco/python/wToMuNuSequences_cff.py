import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.WReco.wToMuNuHLTPaths_cfi import *
wToMuNuHLTSequence = cms.Sequence(wToMuNuHLTFilter)
wToMuNuAnalysis = cms.Sequence(wToMuNuHLTSequence)

