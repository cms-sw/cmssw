import FWCore.ParameterSet.Config as cms

# local clusterizer
from RecoPPS.Local.ctppsPixelClusters_cfi import ctppsPixelClusters

# local rechit producer
from RecoPPS.Local.ctppsPixelRecHits_cfi import ctppsPixelRecHits

# local track producer
from RecoPPS.Local.ctppsPixelLocalTracks_cfi import ctppsPixelLocalTracks



ctppsPixelLocalReconstructionTask = cms.Task(
    ctppsPixelClusters,ctppsPixelRecHits,ctppsPixelLocalTracks
)
ctppsPixelLocalReconstruction = cms.Sequence(ctppsPixelLocalReconstructionTask)
