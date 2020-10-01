import FWCore.ParameterSet.Config as cms

siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0)
)

_siPixelRecHitsPreSplitting = siPixelRecHits.clone(
    src = 'siPixelClustersPreSplitting'
)

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
siPixelRecHitsPreSplitting = SwitchProducerCUDA(
    cpu = _siPixelRecHitsPreSplitting.clone()
)



from Configuration.ProcessModifiers.gpu_cff import gpu
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDA_cfi import siPixelRecHitCUDA as _siPixelRecHitCUDA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromSOA_cfi import siPixelRecHitFromSOA as _siPixelRecHitFromSOA

gpu.toModify(siPixelRecHitsPreSplitting, 
    cuda = _siPixelRecHitFromSOA.clone()
)


siPixelRecHitsPreSplittingTask = cms.Task(siPixelRecHitsPreSplitting)

siPixelRecHitsCUDAPreSplitting = _siPixelRecHitCUDA.clone(
    beamSpot = "offlineBeamSpotToCUDA"
)

siPixelRecHitsLegacyPreSplitting = _siPixelRecHitFromSOA.clone()
siPixelRecHitsPreSplittingTaskCUDA = cms.Task(
    siPixelRecHitsCUDAPreSplitting,
    siPixelRecHitsLegacyPreSplitting,
)

from Configuration.ProcessModifiers.gpu_cff import gpu
_siPixelRecHitsPreSplittingTask_gpu = siPixelRecHitsPreSplittingTask.copy()
_siPixelRecHitsPreSplittingTask_gpu.add(siPixelRecHitsPreSplittingTaskCUDA)
gpu.toReplaceWith(siPixelRecHitsPreSplittingTask, _siPixelRecHitsPreSplittingTask_gpu)
