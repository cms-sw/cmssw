import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis

siPixelDigisTask = cms.Task(
    # SwitchProducer wrapping the legacy pixel digis producer or an alias combining the pixel digis information converted from SoA
    siPixelDigis
)

# copy the pixel digis (except errors) and clusters to the host
from EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi import siPixelDigisSoAFromCUDA as _siPixelDigisSoAFromCUDA
siPixelDigisSoA = _siPixelDigisSoAFromCUDA.clone(
    src = "siPixelClustersPreSplittingCUDA"
)

# copy the pixel digis errors to the host
from EventFilter.SiPixelRawToDigi.siPixelDigiErrorsSoAFromCUDA_cfi import siPixelDigiErrorsSoAFromCUDA as _siPixelDigiErrorsSoAFromCUDA
siPixelDigiErrorsSoA = _siPixelDigiErrorsSoAFromCUDA.clone(
    src = "siPixelClustersPreSplittingCUDA"
)

# convert the pixel digis errors to the legacy format
from EventFilter.SiPixelRawToDigi.siPixelDigiErrorsFromSoA_cfi import siPixelDigiErrorsFromSoA as _siPixelDigiErrorsFromSoA
siPixelDigiErrors = _siPixelDigiErrorsFromSoA.clone()

# use the Phase 1 settings
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigiErrors,
    UsePhase1 = True
)

from Configuration.ProcessModifiers.gpu_cff import gpu
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
(gpu & ~phase2_tracker).toReplaceWith(siPixelDigisTask, cms.Task(
    # copy the pixel digis (except errors) and clusters to the host
    siPixelDigisSoA,
    # copy the pixel digis errors to the host
    siPixelDigiErrorsSoA,
    # convert the pixel digis errors to the legacy format
    siPixelDigiErrors,
    # SwitchProducer wrapping the legacy pixel digis producer or an alias combining the pixel digis information converted from SoA
    siPixelDigisTask.copy()
))

# Remove siPixelDigis until we have phase2 pixel digis
phase2_tracker.toReplaceWith(siPixelDigisTask, cms.Task()) #FIXME
