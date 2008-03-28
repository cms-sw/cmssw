import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.WReco.wToENuHLTPaths_cfi import *
wToENuHLTSequence = cms.Sequence(wToENuHLTFilter)
wToENuAnalysis = cms.Sequence(wToENuHLTSequence)

