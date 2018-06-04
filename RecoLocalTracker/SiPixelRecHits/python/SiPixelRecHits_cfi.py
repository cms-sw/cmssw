import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.gpu_cff import gpu

siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),
)

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitHeterogeneous_cfi import siPixelRecHitHeterogeneous as _siPixelRecHitHeterogeneous
gpu.toReplaceWith(siPixelRecHits, _siPixelRecHitHeterogeneous)

siPixelRecHitsPreSplitting = siPixelRecHits.clone(
    src = 'siPixelClustersPreSplitting'
)
