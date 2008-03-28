import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.WReco.wToENuSequences_cff import *
from ElectroWeakAnalysis.WReco.wToTauNuSequences_cff import *
from ElectroWeakAnalysis.WReco.wToMuNuSequences_cff import *
wReco = cms.Sequence(wToENuAnalysis+wToMuNuAnalysis+wToTauNuAnalysis)

