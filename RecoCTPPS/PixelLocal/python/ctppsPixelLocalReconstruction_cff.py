import FWCore.ParameterSet.Config as cms

# geometry
#from Geometry.VeryForwardGeometry.geometryRP_cfi import *


# local track fitting
from RecoCTPPS.PixelLocal.ctppsPixelClusters_cfi import ctppsPixelClusters

# local rechit producer
from RecoCTPPS.PixelLocal.ctppsPixelRecHits_cfi import ctppsPixelRecHits

ctppsPixelLocalReconstruction = cms.Sequence(
    ctppsPixelClusters*ctppsPixelRecHits
)
