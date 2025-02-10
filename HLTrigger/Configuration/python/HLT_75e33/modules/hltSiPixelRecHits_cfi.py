import FWCore.ParameterSet.Config as cms

hltSiPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    CPE = cms.string('PixelCPEGeneric'),
    src = cms.InputTag("hltSiPixelClusters")
)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toReplaceWith(hltSiPixelRecHits, cms.EDProducer('SiPixelRecHitFromSoAAlpakaPhase2',
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    src = cms.InputTag('hltSiPixelClusters'),
))
