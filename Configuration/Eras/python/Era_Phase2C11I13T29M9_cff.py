import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11I13_cff import Phase2C11I13
from Configuration.Eras.Modifier_phase2_brickedPixels_cff import phase2_brickedPixels
from Configuration.Eras.Modifier_phase2_GE0_cff import phase2_GE0

Phase2C11I13T29M9 = cms.ModifierChain(Phase2C11I13, phase2_brickedPixels, phase2_GE0)
