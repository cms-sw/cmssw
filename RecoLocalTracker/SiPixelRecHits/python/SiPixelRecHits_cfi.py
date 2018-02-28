import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits.SiPixelRecHitsGPU_cfi import siPixelRecHits as _siPixelRecHitsGPU
siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),

)

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toReplaceWith(siPixelRecHits, _siPixelRecHitsGPU)

siPixelRecHitsPreSplitting = siPixelRecHits.clone(
    src = 'siPixelClustersPreSplitting'
)
