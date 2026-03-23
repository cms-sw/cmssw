import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
from Configuration.ProcessModifiers.PixelCPEGeneric_cff import PixelCPEGeneric

Run3_2026 = cms.ModifierChain(Run3_2025, PixelCPEGeneric)
