import FWCore.ParameterSet.Config as cms

#from Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi import *
from Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi import *
from Geometry.VeryForwardGeometryBuilder.ctppsIncludeAlignments_cfi import *
ctppsIncludeAlignments.RealFiles = cms.vstring("Alignment/CTPPS/data/RPixGeometryCorrections.xml")

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer
from RecoCTPPS.PixelLocal.ctppsPixelLocalReconstruction_cff import *

recoCTPPS = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    ctppsPixelLocalReconstruction *
    ctppsLocalTrackLiteProducer
)
