import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_run3_egamma_2022_rereco_cff import run3_egamma_2022_rereco

Run3_2022_rereco = cms.ModifierChain(Run3, run3_egamma_2022_rereco)
