import FWCore.ParameterSet.Config as cms

# geometry
from Geometry.VeryForwardGeometry.geometryRP_cfi import *

# strips reconstruction
from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cfi import totemRPLocalReconstruction

# diamonds reconstruction
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cfi import ctppsDiamondLocalReconstruction

ctppsLocalReconstruction = cms.Sequence(
    totemLocalReconstruction
    * ctppsDiamondLocalReconstruction
)
