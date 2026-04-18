import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_UPC_cff import Run3_2025_UPC
from Configuration.Eras.Modifier_run3_oxygen_cff import run3_oxygen
from Configuration.ProcessModifiers.rpdReco_cff import rpdReco

Run3_2025_UPC_OXY = cms.ModifierChain(Run3_2025_UPC.copyAndExclude([rpdReco]), run3_oxygen)
