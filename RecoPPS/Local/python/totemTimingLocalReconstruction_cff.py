import FWCore.ParameterSet.Config as cms

# reco hit production
from RecoPPS.Local.totemTimingRecHits_cfi import totemTimingRecHits

from RecoPPS.Local.totemTimingLocalTracksSampic_cfi import totemTimingLocalTracksSampic
totemTimingLocalTracksSampic.recHitsTag=cms.InputTag("totemTimingRecHits")
totemTimingLocalReconstructionTaskSampic = cms.Task(
		totemTimingRecHits,
		totemTimingLocalTracksSampic)
totemTimingLocalReconstructionSampic = cms.Sequence(totemTimingLocalReconstructionTaskSampic)
	

from RecoPPS.Local.totemTimingLocalTracks_cfi import totemTimingLocalTracks
totemTimingLocalTracks.recHitsTag=cms.InputTag("totemTimingRecHits")
totemTimingLocalReconstructionTask = cms.Task(
		totemTimingRecHits,
		totemTimingLocalTracks)
totemTimingLocalReconstruction = cms.Sequence(totemTimingLocalReconstructionTask)
