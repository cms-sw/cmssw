import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
from Configuration.Eras.Modifier_phase2_3DPixels_cff import phase2_3DPixels

Phase2C11T23 = cms.ModifierChain(Phase2C11, phase2_3DPixels)
