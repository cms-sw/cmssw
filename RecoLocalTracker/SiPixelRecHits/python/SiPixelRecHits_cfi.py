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
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDA_cfi import siPixelRecHitFromCUDA as _siPixelRecHitFromCUDA

gpu.toModify(siPixelRecHitsPreSplitting, 
    cuda = _siPixelRecHitFromCUDA.clone()
)


siPixelRecHitsPreSplittingTask = cms.Task(siPixelRecHitsPreSplitting)

siPixelRecHitsPreSplittingCUDA = _siPixelRecHitCUDA.clone(
    beamSpot = "offlineBeamSpotToCUDA"
)

siPixelRecHitsPreSplittingLegacy = _siPixelRecHitFromCUDA.clone()
siPixelRecHitsPreSplittingTaskCUDA = cms.Task(
    siPixelRecHitsPreSplittingCUDA,
    siPixelRecHitsPreSplittingLegacy,
)

from Configuration.ProcessModifiers.gpu_cff import gpu
_siPixelRecHitsPreSplittingTask_gpu = siPixelRecHitsPreSplittingTask.copy()
_siPixelRecHitsPreSplittingTask_gpu.add(siPixelRecHitsPreSplittingTaskCUDA)
gpu.toReplaceWith(siPixelRecHitsPreSplittingTask, _siPixelRecHitsPreSplittingTask_gpu)
