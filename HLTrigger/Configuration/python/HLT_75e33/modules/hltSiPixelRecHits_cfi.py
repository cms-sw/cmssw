import FWCore.ParameterSet.Config as cms

hltSiPixelRecHits = cms.EDProducer('SiPixelRecHitFromSoAAlpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    src = cms.InputTag('hltSiPixelClusters'),
)
