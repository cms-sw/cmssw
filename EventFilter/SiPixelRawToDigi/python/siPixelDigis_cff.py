import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis

siPixelDigisTask = cms.Task(
    # SwitchProducer wrapping the legacy pixel digis producer or an alias combining the pixel digis information converted from SoA
    siPixelDigis
)

# Phase 2 Tracker Modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
# Remove siPixelDigis until we have phase2 pixel digis
phase2_tracker.toReplaceWith(siPixelDigisTask, cms.Task()) #FIXME
