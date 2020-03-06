import FWCore.ParameterSet.Config as cms

# clusterization
from RecoPPS.Local.totemRPClusterProducer_cfi import *

# reco hit production
from RecoPPS.Local.totemRPRecHitProducer_cfi import *

# non-parallel pattern recognition
from RecoPPS.Local.totemRPUVPatternFinder_cfi import *

# local track fitting
from RecoPPS.Local.totemRPLocalTrackFitter_cfi import *

totemRPLocalReconstructionTask = cms.Task(
    totemRPClusterProducer ,
    totemRPRecHitProducer ,
    totemRPUVPatternFinder ,
    totemRPLocalTrackFitter
)
totemRPLocalReconstruction = cms.Sequence(totemRPLocalReconstructionTask)
