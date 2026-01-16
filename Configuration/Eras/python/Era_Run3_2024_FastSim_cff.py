import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM

Run3_2024_FastSim = Run3_2024.copyAndExclude([run3_GEM])
