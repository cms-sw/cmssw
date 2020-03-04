import FWCore.ParameterSet.Config as cms

# local clusterizer
from RecoCTPPS.PixelLocal.ctppsPixelClusters_cfi import ctppsPixelClusters

# local rechit producer
from RecoCTPPS.PixelLocal.ctppsPixelRecHits_cfi import ctppsPixelRecHits

# local track producer
from RecoCTPPS.PixelLocal.ctppsPixelLocalTracks_cfi import ctppsPixelLocalTracks

ctppsPixelLocalReconstructionTask = cms.Task(
    ctppsPixelClusters,ctppsPixelRecHits,ctppsPixelLocalTracks
)
ctppsPixelLocalReconstruction = cms.Sequence(ctppsPixelLocalReconstructionTask)
