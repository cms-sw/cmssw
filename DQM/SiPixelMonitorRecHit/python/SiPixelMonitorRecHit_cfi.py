import FWCore.ParameterSet.Config as cms

SiPixelRecHitSource = cms.EDFilter("SiPixelRecHitSource",
    src = cms.InputTag("siPixelRecHits"),
    outputFile = cms.string('Pixel_DQM_RecHits.root'),
    saveFile = cms.untracked.bool(False)
)


