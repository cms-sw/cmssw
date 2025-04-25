import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2025_cff import Run3_2025
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Run3_2025_DDD = Run3_2025.copyAndExclude([dd4hep])

