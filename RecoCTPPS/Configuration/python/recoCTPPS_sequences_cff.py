import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.totemTimingLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer
from RecoCTPPS.PixelLocal.ctppsPixelLocalReconstruction_cff import *

from Geometry.VeryForwardGeometryBuilder.ctppsIncludeAlignmentsFromXML_cfi import *
ctppsIncludeAlignmentsFromXML.RealFiles = cms.vstring("Alignment/CTPPS/data/RPixGeometryCorrections.xml")

recoCTPPSdets = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    totemTimingLocalReconstruction *
    ctppsPixelLocalReconstruction *
    ctppsLocalTrackLiteProducer
)
