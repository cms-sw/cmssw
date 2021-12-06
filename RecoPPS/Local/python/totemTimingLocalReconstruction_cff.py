import FWCore.ParameterSet.Config as cms

# reco hit production
from RecoPPS.Local.totemTimingRecHits_cfi import totemTimingRecHits

#Diamond Sampic reconstruction flow
from RecoPPS.Local.diamondSampicLocalTracks_cfi import diamondSampicLocalTracks
diamondSampicLocalTracks.recHitsTag=cms.InputTag("totemTimingRecHits")
diamondSampicLocalReconstructionTask = cms.Task(
		totemTimingRecHits,
		diamondSampicLocalTracks)
diamondSampicLocalReconstruction = cms.Sequence(diamondSampicLocalReconstructionTask)
	
#Original UFSD reconstruction flow
from RecoPPS.Local.totemTimingLocalTracks_cfi import totemTimingLocalTracks
totemTimingLocalTracks.recHitsTag=cms.InputTag("totemTimingRecHits")
totemTimingLocalReconstructionTask = cms.Task(
		totemTimingRecHits,
		totemTimingLocalTracks)
totemTimingLocalReconstruction = cms.Sequence(totemTimingLocalReconstructionTask)
