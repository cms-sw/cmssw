import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.gpu_cff import gpu

siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),
)

from RecoLocalTracker.SiPixelRecHits.SiPixelRecHitsGPU_cfi import siPixelRecHits as _siPixelRecHitsGPU
gpu.toReplaceWith(siPixelRecHits, _siPixelRecHitsGPU)

siPixelRecHitsPreSplitting = siPixelRecHits.clone(
    src = 'siPixelClustersPreSplitting'
)
gpu.toModify(siPixelRecHitsPreSplitting, src = 'siPixelDigis')
