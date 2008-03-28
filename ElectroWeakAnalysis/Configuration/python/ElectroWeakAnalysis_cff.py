import FWCore.ParameterSet.Config as cms

#
# ElectroWeakAnalysis standard sequences
#
from ElectroWeakAnalysis.ZReco.ZReco_cff import *
from ElectroWeakAnalysis.WReco.WReco_cff import *
electroWeakAnalysis = cms.Sequence(zReco*wReco)

