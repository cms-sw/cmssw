import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
from Configuration.ProcessModifiers.miniAOD_skip_probQXY_cff import miniAOD_skip_probQXY

# This modifier is for additional settings to run miniAOD on top of 
# ultra-legacy (during LS2) Run-2 AOD produced prior to the Winter22
# campaign where AOD event content was extended

run2_miniAOD_UL_preWinter22 = cms.ModifierChain(run2_miniAOD_UL, miniAOD_skip_probQXY)
