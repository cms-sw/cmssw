import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from Configuration.ProcessModifiers.gpu_cff import gpu

# legacy pixel rechit producer
siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0)
)

# SwitchProducer wrapping the legacy pixel rechit producer
siPixelRecHitsPreSplitting = SwitchProducerCUDA(
    cpu = siPixelRecHits.clone(
        src = 'siPixelClustersPreSplitting'
    )
)

# convert the pixel rechits from legacy to SoA format
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacy_cfi import siPixelRecHitSoAFromLegacy as siPixelRecHitsPreSplittingSoA

siPixelRecHitsPreSplittingTask = cms.Task(
    # SwitchProducer wrapping the legacy pixel rechit producer
    siPixelRecHitsPreSplitting,
    # convert the pixel rechits from legacy to SoA format
    siPixelRecHitsPreSplittingSoA
)

# reconstruct the pixel rechits on the gpu
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDA_cfi import siPixelRecHitCUDA as _siPixelRecHitCUDA
siPixelRecHitsPreSplittingCUDA = _siPixelRecHitCUDA.clone(
    beamSpot = "offlineBeamSpotToCUDA"
)

# transfer the pixel rechits to the host and convert them from SoA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDA_cfi import siPixelRecHitFromCUDA as _siPixelRecHitFromCUDA
gpu.toModify(siPixelRecHitsPreSplitting, 
    cuda = _siPixelRecHitFromCUDA.clone()
)

gpu.toReplaceWith(siPixelRecHitsPreSplittingTask, cms.Task(
    # reconstruct the pixel rechits on the gpu
    siPixelRecHitsPreSplittingCUDA,
    # SwitchProducer wrapping the legacy pixel rechit producer or the transfer of the pixel rechits to the host and the conversion from SoA
    siPixelRecHitsPreSplittingTask.copy()
))
