import FWCore.ParameterSet.Config as cms

# geometry
#from Geometry.VeryForwardGeometry.geometryRP_cfi import *


# local track fitting
from RecoCTPPS.PixelLocal.ctppsPixelClusters_cfi import ctppsPixelClusters

ctppsPixelLocalReconstruction = cms.Sequence(
    ctppsPixelClusters
)
