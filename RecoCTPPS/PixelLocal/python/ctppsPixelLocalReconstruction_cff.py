import FWCore.ParameterSet.Config as cms

# geometry
from Geometry.VeryForwardGeometry.geometryRP_cfi import *
#from Geometry.VeryForwardGeometry.geometry_CTPPS_alaTotem_RECO_cfi import *

# local clusterizer
from RecoCTPPS.PixelLocal.ctppsPixelClusters_cfi import ctppsPixelClusters

# local rechit producer
from RecoCTPPS.PixelLocal.ctppsPixelRecHits_cfi import ctppsPixelRecHits

# local track producer
from RecoCTPPS.PixelLocal.ctppsPixelTracks_cfi import ctppsPixelTracks

#ctppsPixelTracks = cms.EDProducer('CTPPSPixelLocalTrackProducer',
#  patterFinderAlgorithm = cms.string('testPatternAlgorithm')
#)

ctppsPixelLocalReconstruction = cms.Sequence(
    ctppsPixelClusters*ctppsPixelRecHits*ctppsPixelTracks
)
