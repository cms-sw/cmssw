import FWCore.ParameterSet.Config as cms

from RecoPPS.Local.ctppsLocalTrackLiteDefaultProducer_cfi import ctppsLocalTrackLiteDefaultProducer

ctppsLocalTrackLiteProducer = ctppsLocalTrackLiteDefaultProducer.clone()

# enable the module for CTPPS era(s)
from Configuration.Eras.Modifier_ctpps_cff import ctpps
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017

ctpps.toModify(
    ctppsLocalTrackLiteProducer,
    includeStrips = False,
    includeDiamonds = True,
    includePixels = True
)

ctpps_2016.toModify(
    ctppsLocalTrackLiteProducer,
    includeStrips = True,
    includeDiamonds = False,
    includePixels = False
)

ctpps_2017.toModify(
    ctppsLocalTrackLiteProducer,
    includeStrips = True,
    includeDiamonds = True,
    includePixels = True
)

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy

run2_miniAOD_80XLegacy.toModify(ctppsLocalTrackLiteProducer, tagPixelTrack = "" )
