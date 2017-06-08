import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cfi import ctppsLocalTrackLiteProducer
from RecoCTPPS.CTPPSPixelLocal.ctppsPixelLocalReconstruction_cff import *

recoCTPPS = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    ctppsPixelLocalReconstruction *
    ctppsLocalTrackLiteProducer
)
