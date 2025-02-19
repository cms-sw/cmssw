import FWCore.ParameterSet.Config as cms

from HeavyFlavorAnalysis.Skimming.tauTo3MuHLTPath_cfi import *
from HeavyFlavorAnalysis.Skimming.tauTo3MuFilter_cfi import *
tauTo3MuSkim = cms.Sequence(tauTo3MuHLTFilter+tauTo3MuFilter)

