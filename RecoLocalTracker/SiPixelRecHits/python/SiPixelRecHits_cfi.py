import FWCore.ParameterSet.Config as cms

siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),

)

from RecoLocalTracker.SiPixelRecHits.SiPixelRecHitsGPU_cfi import siPixelRecHits as _siPixelRecHitsGPU
from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toReplaceWith(siPixelRecHits, _siPixelRecHitsGPU)

siPixelRecHitsPreSplitting = siPixelRecHits.clone(
    src = 'siPixelClustersPreSplitting'
)
