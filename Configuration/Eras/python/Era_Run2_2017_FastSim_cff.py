import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017

Run2_2017_FastSim = Run2_2017.copyAndExclude([run2_GEM_2017])
