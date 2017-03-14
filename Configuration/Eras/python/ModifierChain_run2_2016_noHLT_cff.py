import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.Modifier_HLT_2016_cff import HLT_2016

run2_2016_noHLT = Run2_2016.copyAndExclude([HLT_2016])
