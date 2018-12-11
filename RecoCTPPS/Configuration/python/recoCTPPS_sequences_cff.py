import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.totemTimingLocalReconstruction_cff import *
from RecoCTPPS.PixelLocal.ctppsPixelLocalReconstruction_cff import *

from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer

from RecoCTPPS.ProtonReconstruction.ctppsProtonReconstruction_cfi import *

# TODO: load these data from DB
from CondFormats.CTPPSReadoutObjects.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring("Alignment/CTPPS/data/RPixGeometryCorrections.xml")
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles += cms.vstring("Validation/CTPPS/test/year_2016/alignment_export_2018_12_07.1.xml")

# TODO: load these data from DB
from Validation.CTPPS.year_2016.ctppsLHCInfoESSource_cfi import *

# TODO: load these data from DB
from Validation.CTPPS.year_2016.ctppsOpticalFunctionsESSource_cfi import *

recoCTPPSdets = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    totemTimingLocalReconstruction *
    ctppsPixelLocalReconstruction *
    ctppsLocalTrackLiteProducer *
    ctppsProtonReconstruction
)
