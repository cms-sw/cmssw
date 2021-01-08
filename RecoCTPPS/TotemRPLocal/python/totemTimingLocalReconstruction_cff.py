import FWCore.ParameterSet.Config as cms

# reco hit production
from RecoCTPPS.TotemRPLocal.totemTimingRecHits_cfi import totemTimingRecHits

# local track fitting
from RecoCTPPS.TotemRPLocal.totemTimingLocalTracks_cfi import totemTimingLocalTracks

totemTimingLocalReconstructionTask = cms.Task(
    totemTimingRecHits ,
    totemTimingLocalTracks
)
totemTimingLocalReconstruction = cms.Sequence(totemTimingLocalReconstructionTask)
