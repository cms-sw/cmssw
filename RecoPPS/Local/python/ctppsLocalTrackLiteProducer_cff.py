import FWCore.ParameterSet.Config as cms

from RecoPPS.Local.ctppsLocalTrackLiteDefaultProducer_cfi import ctppsLocalTrackLiteDefaultProducer

ctppsLocalTrackLiteProducer = ctppsLocalTrackLiteDefaultProducer.clone()

# enable the module for CTPPS era(s)
from Configuration.Eras.Modifier_ctpps_cff import ctpps
ctpps.toModify(
    ctppsLocalTrackLiteProducer,
    includeStrips = True,
    includeDiamonds = True,
    includePixels = True
)

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy

run2_miniAOD_80XLegacy.toModify(ctppsLocalTrackLiteProducer, tagPixelTrack = "" )
