import FWCore.ParameterSet.Config as cms

# reco hit production
from RecoPPS.Local.ctppsDiamondRecHits_cfi import ctppsDiamondRecHits

# local track fitting
from RecoPPS.Local.ctppsDiamondLocalTracks_cfi import ctppsDiamondLocalTracks

ctppsDiamondLocalReconstructionTask = cms.Task(
    ctppsDiamondRecHits,
    ctppsDiamondLocalTracks
)
ctppsDiamondLocalReconstruction = cms.Sequence(ctppsDiamondLocalReconstructionTask)
