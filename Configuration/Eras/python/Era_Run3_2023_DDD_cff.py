import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2023_cff import Run3_2023
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Run3_2023_DDD = Run3_2023.copyAndExclude([dd4hep])

