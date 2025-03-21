import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis

siPixelDigisTask = cms.Task(
    siPixelDigis
)

# FIXME remove siPixelDigis until we have Phase 2 pixel digis
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(siPixelDigisTask, cms.Task())
