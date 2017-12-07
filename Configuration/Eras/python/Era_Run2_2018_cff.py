import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
from Configuration.Eras.Modifier_run2_CSC_2018_cff import run2_CSC_2018

Run2_2018 = cms.ModifierChain(Run2_2017.copyAndExclude([run2_HEPlan1_2017]), run2_CSC_2018)


