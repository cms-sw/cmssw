import FWCore.ParameterSet.Config as cms

# local clusterizer
from RecoPPS.Local.ctppsPixelClusters_cfi import ctppsPixelClusters

# local rechit producer
from RecoPPS.Local.ctppsPixelRecHits_cfi import ctppsPixelRecHits

# local track producer
from RecoPPS.Local.ctppsPixelLocalTracks_cfi import ctppsPixelLocalTracks

#from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
#from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
#from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
#(ctpps_2016 | ctpps_2017 | ctpps_2018).toModify(ctppsPixelLocalTracks, isBadPot = cms.bool(False))

ctppsPixelLocalReconstructionTask = cms.Task(
    ctppsPixelClusters,ctppsPixelRecHits,ctppsPixelLocalTracks
)
ctppsPixelLocalReconstruction = cms.Sequence(ctppsPixelLocalReconstructionTask)
