import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_UPC_cff import Run3_2025_UPC
from Configuration.Eras.Modifier_run3_oxygen_cff import run3_oxygen

Run3_2025_UPC_OXY = cms.ModifierChain(Run3_2025_UPC, run3_oxygen)
