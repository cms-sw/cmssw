import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
from Configuration.Eras.Util_fastSimPhase2_cff import fastSimPhase2

Phase2C17I13M9_FastSim = fastSimPhase2(Phase2C17I13M9)
