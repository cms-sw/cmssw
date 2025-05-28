import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM

Run3_2025_FastSim = Run3_2025.copyAndExclude([run3_GEM])
