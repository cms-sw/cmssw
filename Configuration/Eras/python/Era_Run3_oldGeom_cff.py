import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_run3_geomOld_cff import run3_geomOld

Run3_oldGeom = cms.ModifierChain(Run3, run3_geomOld)
