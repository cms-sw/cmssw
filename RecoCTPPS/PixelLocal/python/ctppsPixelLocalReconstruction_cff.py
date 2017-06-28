import FWCore.ParameterSet.Config as cms

# geometry
#from Geometry.VeryForwardGeometry.geometryRP_cfi import *


# local track fitting
from RecoCTPPS.PixelLocal.CTPPSPixelClusterizer_cfi import clusterProd

ctppsPixelLocalReconstruction = cms.Sequence(
    clusterProd
)
