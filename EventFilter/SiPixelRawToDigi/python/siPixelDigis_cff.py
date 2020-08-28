import FWCore.ParameterSet.Config as cms

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis
from EventFilter.SiPixelRawToDigi.siPixelDigisSoAFromCUDA_cfi import siPixelDigisSoAFromCUDA as _siPixelDigisSoAFromCUDA
from EventFilter.SiPixelRawToDigi.siPixelDigiErrorsSoAFromCUDA_cfi import siPixelDigiErrorsSoAFromCUDA as _siPixelDigiErrorsSoAFromCUDA
from EventFilter.SiPixelRawToDigi.siPixelDigiErrorsFromSoA_cfi import siPixelDigiErrorsFromSoA as _siPixelDigiErrorsFromSoA

siPixelDigisTask = cms.Task(siPixelDigis)

siPixelDigisSoA = _siPixelDigisSoAFromCUDA.clone(
    src = "siPixelClustersCUDAPreSplitting"
)
siPixelDigiErrorsSoA = _siPixelDigiErrorsSoAFromCUDA.clone(
    src = "siPixelClustersCUDAPreSplitting"
)
siPixelDigiErrors = _siPixelDigiErrorsFromSoA.clone()

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigiErrors, UsePhase1=True)

siPixelDigisTaskCUDA = cms.Task(
    siPixelDigisSoA,
    siPixelDigiErrorsSoA,
    siPixelDigiErrors
)

from Configuration.ProcessModifiers.gpu_cff import gpu
_siPixelDigisTask_gpu = siPixelDigisTask.copy()
_siPixelDigisTask_gpu.add(siPixelDigisTaskCUDA)
gpu.toReplaceWith(siPixelDigisTask, _siPixelDigisTask_gpu)
