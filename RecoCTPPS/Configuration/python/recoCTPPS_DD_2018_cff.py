import FWCore.ParameterSet.Config as cms

from Geometry.VeryForwardGeometry.geometryCTPPSFromDD_2018_cfi import *

#from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
#from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
#from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer
from RecoCTPPS.PixelLocal.ctppsPixelLocalReconstruction_cff import *

recoCTPPS = cms.Sequence(
    ctppsPixelLocalReconstruction
)


