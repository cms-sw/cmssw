import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_run2_HE_2017_cff import run2_HE_2017

Run2_2018 = cms.ModifierChain(Run2_2017, run2_HE_2017)

