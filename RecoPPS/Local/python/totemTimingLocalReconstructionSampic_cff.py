import FWCore.ParameterSet.Config as cms

# reco hit production
from RecoPPS.Local.totemTimingRecHits_cfi import totemTimingRecHits

# local track fitting
from RecoPPS.Local.totemTimingLocalTracksSampic_cfi import totemTimingLocalTracksSampic
totemTimingLocalTracksSampic.recHitsTag=cms.InputTag("totemTimingRecHits")
totemTimingLocalReconstructionTask = cms.Task(
    totemTimingRecHits,
    totemTimingLocalTracksSampic
)
totemTimingLocalReconstruction = cms.Sequence(totemTimingLocalReconstructionTask)
