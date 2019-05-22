import FWCore.ParameterSet.Config as cms
import EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi

siPixelDigis = EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi.siPixelRawToDigi.clone()

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigis, UsePhase1=True)
