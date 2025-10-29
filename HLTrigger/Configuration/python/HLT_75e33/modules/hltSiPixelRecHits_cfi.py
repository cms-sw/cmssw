import FWCore.ParameterSet.Config as cms

hltSiPixelRecHits = cms.EDProducer('SiPixelRecHitFromSoAAlpakaPhase2',
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    src = cms.InputTag('hltSiPixelClusters'),
)
