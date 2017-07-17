import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM

Run3 = cms.ModifierChain(Run2_2017, run3_GEM)

