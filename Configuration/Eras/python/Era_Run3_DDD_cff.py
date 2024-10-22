import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

Run3_DDD = Run3.copyAndExclude([dd4hep])

