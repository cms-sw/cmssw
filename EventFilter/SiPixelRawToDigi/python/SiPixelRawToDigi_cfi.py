import FWCore.ParameterSet.Config as cms
import EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi
import EventFilter.SiPixelRawToDigi.siPixelRawToDigiGPU_cfi

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

siPixelDigisGPU = EventFilter.SiPixelRawToDigi.siPixelRawToDigiGPU_cfi.siPixelRawToDigiGPU.clone()
siPixelDigisGPU.Timing = cms.untracked.bool(False)
siPixelDigisGPU.IncludeErrors = cms.bool(True)
siPixelDigisGPU.InputLabel = cms.InputTag("rawDataCollector")
siPixelDigisGPU.UseQualityInfo = cms.bool(False)
## ErrorList: list of error codes used by tracking to invalidate modules
siPixelDigisGPU.ErrorList = cms.vint32(29)
## UserErrorList: list of error codes used by Pixel experts for investigation
siPixelDigisGPU.UserErrorList = cms.vint32(40)
##  Use pilot blades
siPixelDigisGPU.UsePilotBlade = cms.bool(False)
##  Use phase1
siPixelDigisGPU.UsePhase1 = cms.bool(False)
## Empty Regions PSet means complete unpacking
siPixelDigisGPU.Regions = cms.PSet( )
siPixelDigisGPU.CablingMapLabel = cms.string("")

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigis, UsePhase1=True)
phase1Pixel.toModify(siPixelDigisGPU, UsePhase1=True)

