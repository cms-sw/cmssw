import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

siPixelRawData = cms.EDProducer("SiPixelDigiToRaw",
    Timing = cms.untracked.bool(False),
    InputLabel = cms.InputTag("simSiPixelDigis"),
##  Use phase1
    UsePhase1 = cms.bool(False),
)
eras.phase1Pixel.toModify(siPixelRawData, UsePhase1=True)


