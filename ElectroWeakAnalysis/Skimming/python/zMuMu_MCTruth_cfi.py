import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.Skimming.dimuonsHLTFilter_cfi import *
from ElectroWeakAnalysis.Skimming.mcTruthForDimuons_cff import *

dimuonsMCTruth = cms.Path(dimuonsHLTFilter+
                          mcTruthForDimuons
)

mcEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
    ### MC matching infos
    'keep *_genParticles_*_*',
    'keep *_allDimuonsMCMatch_*_*',
    )
)


