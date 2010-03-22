import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.Skimming.dimuonsHLTFilter_cfi import *
from ElectroWeakAnalysis.Skimming.mcTruthForDimuons_cff import *

dimuonsMCTruth = cms.Path(dimuonsHLTFilter+
                          mcTruthForDimuons
)


