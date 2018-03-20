import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer
from RecoCTPPS.PixelLocal.ctppsPixelLocalReconstruction_cff import *

from Geometry.VeryForwardGeometryBuilder.ctppsIncludeAlignments_cfi import *
ctppsIncludeAlignments.RealFiles = cms.vstring("Alignment/CTPPS/data/RPixGeometryCorrections.xml")

recoCTPPSdets = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    ctppsPixelLocalReconstruction *
    ctppsLocalTrackLiteProducer
)
