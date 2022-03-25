import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM

Run3_FastSim = Run3.copyAndExclude([run3_GEM])
