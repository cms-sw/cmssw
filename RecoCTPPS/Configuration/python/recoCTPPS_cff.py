import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer

recoCTPPS = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    ctppsLocalTrackLiteProducer
)
