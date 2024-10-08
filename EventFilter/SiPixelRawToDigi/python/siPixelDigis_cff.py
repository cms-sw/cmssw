import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis

siPixelDigisTask = cms.Task(
    # legacy pixel digis producer or an alias combining the pixel digis information converted from SoA
    siPixelDigis
)

# remove siPixelDigis until we have phase2 pixel digis
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(siPixelDigisTask, cms.Task()) #FIXME
