import FWCore.ParameterSet.Config as cms

from RecoCTPPS.TotemRPLocal.totemRPLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.totemTimingLocalReconstruction_cff import *
from RecoCTPPS.TotemRPLocal.ctppsLocalTrackLiteProducer_cff import ctppsLocalTrackLiteProducer
from RecoCTPPS.PixelLocal.ctppsPixelLocalReconstruction_cff import *

from CondFormats.CTPPSReadoutObjects.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi import *
ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring("Alignment/CTPPS/data/RPixGeometryCorrections.xml")

#--- temporary workaround while calibration conditions are integrated within the DB
from CondFormats.CTPPSReadoutObjects.ppsTimingCalibrationESSource_cfi import ppsTimingCalibrationESSource
ppsTimingCalibrationESSource.calibrationFile = cms.FileInPath('RecoCTPPS/TotemRPLocal/data/timing_offsets_ufsd_2018.dec18.cal.json')
#--- to be removed before the integration

recoCTPPSdets = cms.Sequence(
    totemRPLocalReconstruction *
    ctppsDiamondLocalReconstruction *
    totemTimingLocalReconstruction *
    ctppsPixelLocalReconstruction *
    ctppsLocalTrackLiteProducer
)
