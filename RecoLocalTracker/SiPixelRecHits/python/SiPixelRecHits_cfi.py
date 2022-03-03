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
siPixelRecHitsPreSplittingSoA = _siPixelRecHitsPreSplittingSoA.clone(convertToLegacy=True)
# phase 2 tracker modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(siPixelRecHitsPreSplittingSoA,
    isPhase2 = True)
# modifier used to prompt patatrack pixel tracks reconstruction on cpu
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
pixelNtupletFit.toModify(siPixelRecHitsPreSplitting,
    cpu = siPixelRecHitsPreSplittingSoA.clone()
)

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

(gpu & pixelNtupletFit).toModify(siPixelRecHitsPreSplitting,
    cpu = cms.EDAlias(
            siPixelRecHitsPreSplittingSoA = cms.VPSet(
                 cms.PSet(type = cms.string("SiPixelRecHitedmNewDetSetVector")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )
         ),
    cuda = _siPixelRecHitFromCUDA.clone())

(gpu & pixelNtupletFit).toReplaceWith(siPixelRecHitsPreSplittingTask, cms.Task(
    # reconstruct the pixel rechits on the gpu
    siPixelRecHitsPreSplittingCUDA,
    # producing and converting on cpu
    siPixelRecHitsPreSplittingSoA,
    # SwitchProducer wrapping an EDAlias on cpu or the converter from SoA to legacy on gpu
    siPixelRecHitsPreSplittingTask.copy()
))
