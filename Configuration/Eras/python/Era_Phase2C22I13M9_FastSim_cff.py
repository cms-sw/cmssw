import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
from Configuration.Eras.Util_fastSimPhase2_cff import fastSimPhase2

# fastSimPhase2 excludes enableTruth: the truth accumulator reads full-sim g4SimHits
# during classic mixing, which FastSim does not provide.
Phase2C22I13M9_FastSim = fastSimPhase2(Phase2C22I13M9)
