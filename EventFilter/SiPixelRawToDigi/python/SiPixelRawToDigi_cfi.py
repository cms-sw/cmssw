import FWCore.ParameterSet.Config as cms
import EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi
import RecoLocalTracker.SiPixelClusterizer.siPixelDigiHeterogeneousConverter_cfi

siPixelDigis = EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi.siPixelRawToDigi.clone()
siPixelDigis.Timing = cms.untracked.bool(False)
siPixelDigis.IncludeErrors = cms.bool(True)
siPixelDigis.InputLabel = cms.InputTag("siPixelRawData")
siPixelDigis.UseQualityInfo = cms.bool(False)
## ErrorList: list of error codes used by tracking to invalidate modules
siPixelDigis.ErrorList = cms.vint32(29)
## UserErrorList: list of error codes used by Pixel experts for investigation
siPixelDigis.UserErrorList = cms.vint32(40)
##  Use pilot blades
siPixelDigis.UsePilotBlade = cms.bool(False)
##  Use phase1
siPixelDigis.UsePhase1 = cms.bool(False)
## Empty Regions PSet means complete unpacking
siPixelDigis.Regions = cms.PSet( ) 
siPixelDigis.CablingMapLabel = cms.string("")

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigis, UsePhase1=True)

_siPixelDigis_gpu = RecoLocalTracker.SiPixelClusterizer.siPixelDigiHeterogeneousConverter_cfi.siPixelDigiHeterogeneousConverter.clone()
_siPixelDigis_gpu.includeErrors = cms.bool(True)

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toReplaceWith(siPixelDigis, _siPixelDigis_gpu)

