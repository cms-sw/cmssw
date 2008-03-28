import FWCore.ParameterSet.Config as cms

pixelFreeHits = cms.EDFilter("SiPixelRecHitRemover",
    HitCollectionLabel = cms.string('siPixelRecHits'),
    removeHitsList = cms.vstring('')
)


