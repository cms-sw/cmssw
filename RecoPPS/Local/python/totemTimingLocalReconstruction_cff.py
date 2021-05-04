import FWCore.ParameterSet.Config as cms

# reco hit production
from RecoPPS.Local.totemTimingRecHits_cfi import totemTimingRecHits

# local track fitting
from RecoPPS.Local.totemTimingLocalTracks_cfi import totemTimingLocalTracks

totemTimingLocalReconstructionTask = cms.Task(
    totemTimingRecHits ,
    totemTimingLocalTracks
)
totemTimingLocalReconstruction = cms.Sequence(totemTimingLocalReconstructionTask)
