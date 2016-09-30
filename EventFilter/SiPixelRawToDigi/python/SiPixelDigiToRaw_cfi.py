import FWCore.ParameterSet.Config as cms

siPixelRawData = cms.EDProducer("SiPixelDigiToRaw",
    Timing = cms.untracked.bool(False),
    InputLabel = cms.InputTag("simSiPixelDigis"),
##  Use phase1
    UsePhase1 = cms.bool(False),
)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelRawData, UsePhase1=True)


