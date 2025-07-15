import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Run3_2024_DDD = Run3_2024.copyAndExclude([dd4hep])

