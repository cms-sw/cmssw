import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.totemTimingLocalReconstruction_cff import *
from RecoCTPPS.PixelLocal.ctppsPixelLocalReconstruction_cff import *

from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer

from RecoCTPPS.ProtonReconstruction.ctppsProtons_cff import *

from Geometry.VeryForwardGeometry.geometryRPFromDB_cfi import *

recoCTPPS = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    totemTimingLocalReconstruction *
    ctppsPixelLocalReconstruction *
    ctppsLocalTrackLiteProducer *
    ctppsProtons
)
