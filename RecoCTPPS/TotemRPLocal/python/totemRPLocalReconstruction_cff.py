import FWCore.ParameterSet.Config as cms

# clusterization
from RecoCTPPS.TotemRPLocal.totemRPClusterProducer_cfi import *

# reco hit production
from RecoCTPPS.TotemRPLocal.totemRPRecHitProducer_cfi import *

# non-parallel pattern recognition
from RecoCTPPS.TotemRPLocal.totemRPUVPatternFinder_cfi import *

# local track fitting
from RecoCTPPS.TotemRPLocal.totemRPLocalTrackFitter_cfi import *

totemRPLocalReconstructionTask = cms.Task(
    totemRPClusterProducer ,
    totemRPRecHitProducer ,
    totemRPUVPatternFinder ,
    totemRPLocalTrackFitter
)
totemRPLocalReconstruction = cms.Sequence(totemRPLocalReconstructionTask)
