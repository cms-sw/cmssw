import FWCore.ParameterSet.Config as cms

# geometry
#from Geometry.VeryForwardGeometry.geometryRP_cfi import *


# local track fitting
from RecoCTPPS.PixelLocal.CTPPSPixelClusterProducer_cfi import ctppsPixelClusters

ctppsPixelLocalReconstruction = cms.Sequence(
    ctppsPixelClusters
)
