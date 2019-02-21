import FWCore.ParameterSet.Config as cms
import EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi

siPixelDigis = EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi.siPixelRawToDigi.clone()

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigis, UsePhase1=True)

import RecoLocalTracker.SiPixelClusterizer.siPixelDigiHeterogeneousConverter_cfi
_siPixelDigis_gpu = RecoLocalTracker.SiPixelClusterizer.siPixelDigiHeterogeneousConverter_cfi.siPixelDigiHeterogeneousConverter.clone()
_siPixelDigis_gpu.includeErrors = cms.bool(True)

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toReplaceWith(siPixelDigis, _siPixelDigis_gpu)
