import FWCore.ParameterSet.Config as cms

siPixelRecHits = cms.EDProducer("SiPixelRecHitGPU",
    src = cms.InputTag("siPixelDigis"),
    CPE = cms.string('PixelCPEFast'),   # Generic
    VerboseLevel = cms.untracked.int32(0),
)
