import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from Configuration.ProcessModifiers.gpu_cff import gpu

# legacy pixel rechit producer
siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0)
)

from Configuration.Eras.Modifier_phase2_brickedPixels_cff import phase2_brickedPixels
phase2_brickedPixels.toModify(siPixelRecHits,
                              CPE = 'PixelCPEGenericForBricked'
)

# SwitchProducer wrapping the legacy pixel rechit producer
siPixelRecHitsPreSplitting = SwitchProducerCUDA(
    cpu = siPixelRecHits.clone(
        src = 'siPixelClustersPreSplitting'
    )
)

# convert the pixel rechits from legacy to SoA format
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacy_cfi import siPixelRecHitSoAFromLegacy as _siPixelRecHitsPreSplittingSoA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromCUDA_cfi import siPixelRecHitSoAFromCUDA as _siPixelRecHitSoAFromCUDA

siPixelRecHitsPreSplittingCPU = _siPixelRecHitsPreSplittingSoA.clone(convertToLegacy=True)

# phase 2 tracker modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(siPixelRecHitsPreSplittingCPU,
    isPhase2 = True)

# modifier used to prompt patatrack pixel tracks reconstruction on cpu
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
pixelNtupletFit.toModify(siPixelRecHitsPreSplitting,
    cpu = cms.EDAlias(
            siPixelRecHitsPreSplittingCPU = cms.VPSet(
                 cms.PSet(type = cms.string("SiPixelRecHitedmNewDetSetVector")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )
))


siPixelRecHitsPreSplittingTask = cms.Task(
    # SwitchProducer wrapping the legacy pixel rechit producer or the cpu SoA producer
    siPixelRecHitsPreSplitting
)

# reconstruct the pixel rechits on the gpu
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDA_cfi import siPixelRecHitCUDA as _siPixelRecHitCUDA
siPixelRecHitsPreSplittingCUDA = _siPixelRecHitCUDA.clone(
    beamSpot = "offlineBeamSpotToCUDA"
)

# transfer the pixel rechits to the host and convert them from SoA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDA_cfi import siPixelRecHitFromCUDA as _siPixelRecHitFromCUDA

#this is an alias for the SoA on GPU or CPU to be used for DQM
siPixelRecHitsPreSplittingSoA = SwitchProducerCUDA(
    cpu = cms.EDAlias(
            siPixelRecHitsPreSplittingCPU = cms.VPSet(
                 cms.PSet(type = cms.string("cmscudacompatCPUTraitsTrackingRecHit2DHeterogeneous")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
)

(gpu & pixelNtupletFit).toModify(siPixelRecHitsPreSplittingSoA,cuda = _siPixelRecHitSoAFromCUDA.clone())

(gpu & pixelNtupletFit).toModify(siPixelRecHitsPreSplitting, cuda = _siPixelRecHitFromCUDA.clone())

pixelNtupletFit.toReplaceWith(siPixelRecHitsPreSplittingTask, cms.Task(
    cms.Task(
        # reconstruct the pixel rechits on the cpu
        siPixelRecHitsPreSplittingCPU,
        # SwitchProducer wrapping an EDAlias on cpu or the converter from SoA to legacy on gpu
        siPixelRecHitsPreSplittingTask.copy(),
        # producing and converting on cpu (if needed)
        siPixelRecHitsPreSplittingSoA)
        )
        )

(gpu & pixelNtupletFit).toReplaceWith(siPixelRecHitsPreSplittingTask, cms.Task(
    # reconstruct the pixel rechits on the gpu or on the cpu
    # (normally only one of the two is run because only one is consumed from later stages)
    siPixelRecHitsPreSplittingCUDA,
    siPixelRecHitsPreSplittingCPU,
    # SwitchProducer wrapping an EDAlias on cpu or the converter from SoA to legacy on gpu
    siPixelRecHitsPreSplittingTask.copy(),
    # producing and converting on cpu (if needed)
    siPixelRecHitsPreSplittingSoA
))
