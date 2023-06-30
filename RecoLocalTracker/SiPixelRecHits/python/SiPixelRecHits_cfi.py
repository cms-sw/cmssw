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

# phase 2 tracker modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
# HIon modifiers
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

# convert the pixel rechits from legacy to SoA format on CPU
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyPhase1_cfi import siPixelRecHitSoAFromLegacyPhase1 as _siPixelRecHitsPreSplittingSoA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyPhase2_cfi import siPixelRecHitSoAFromLegacyPhase2 as _siPixelRecHitsPreSplittingSoAPhase2
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyHIonPhase1_cfi import siPixelRecHitSoAFromLegacyHIonPhase1 as _siPixelRecHitsPreSplittingSoAHIonPhase1

siPixelRecHitsPreSplittingCPU = _siPixelRecHitsPreSplittingSoA.clone(convertToLegacy=True)
pp_on_AA.toReplaceWith(siPixelRecHitsPreSplittingCPU, _siPixelRecHitsPreSplittingSoAHIonPhase1.clone(convertToLegacy=True, CPE = cms.string('PixelCPEFastHIonPhase1')))
phase2_tracker.toReplaceWith(siPixelRecHitsPreSplittingCPU, _siPixelRecHitsPreSplittingSoAPhase2.clone(convertToLegacy=True, CPE = cms.string('PixelCPEFastPhase2')))

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

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDAPhase1_cfi import siPixelRecHitCUDAPhase1 as _siPixelRecHitCUDAPhase1
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDAPhase2_cfi import siPixelRecHitCUDAPhase2 as _siPixelRecHitCUDAPhase2
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitCUDAHIonPhase1_cfi import siPixelRecHitCUDAHIonPhase1 as _siPixelRecHitCUDAHIonPhase1

siPixelRecHitsPreSplittingCUDA = _siPixelRecHitCUDAPhase1.clone(
    beamSpot = "offlineBeamSpotToCUDA"
)
pp_on_AA.toReplaceWith(siPixelRecHitsPreSplittingCUDA,_siPixelRecHitCUDAHIonPhase1.clone(
    beamSpot = "offlineBeamSpotToCUDA"
))
phase2_tracker.toReplaceWith(siPixelRecHitsPreSplittingCUDA,_siPixelRecHitCUDAPhase2.clone(
    beamSpot = "offlineBeamSpotToCUDA"
))

#this is an alias for the SoA on GPU or CPU to be used for DQM
siPixelRecHitsPreSplittingSoA = SwitchProducerCUDA(
    cpu = cms.EDAlias(
            siPixelRecHitsPreSplittingCPU = cms.VPSet(
                 cms.PSet(type = cms.string("pixelTopologyPhase1TrackingRecHitSoAHost")),
                 cms.PSet(type = cms.string("uintAsHostProduct"))
             )),
)

pp_on_AA.toModify(siPixelRecHitsPreSplittingSoA,
cpu = cms.EDAlias(
        siPixelRecHitsPreSplittingCPU = cms.VPSet(
             cms.PSet(type = cms.string("pixelTopologyHIonPhase1TrackingRecHitSoAHost")),
             cms.PSet(type = cms.string("uintAsHostProduct"))
         )))

phase2_tracker.toModify(siPixelRecHitsPreSplittingSoA,
cpu = cms.EDAlias(
        siPixelRecHitsPreSplittingCPU = cms.VPSet(
             cms.PSet(type = cms.string("pixelTopologyPhase2TrackingRecHitSoAHost")),
             cms.PSet(type = cms.string("uintAsHostProduct"))
         )))

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromCUDAPhase1_cfi import siPixelRecHitSoAFromCUDAPhase1 as _siPixelRecHitSoAFromCUDA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromCUDAHIonPhase1_cfi import siPixelRecHitSoAFromCUDAHIonPhase1 as _siPixelRecHitSoAFromCUDAHIonPhase1
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromCUDAPhase2_cfi import siPixelRecHitSoAFromCUDAPhase2 as _siPixelRecHitSoAFromCUDAPhase2

(gpu & pixelNtupletFit).toModify(siPixelRecHitsPreSplittingSoA, cuda = _siPixelRecHitSoAFromCUDA.clone())
(gpu & pixelNtupletFit & phase2_tracker).toModify(siPixelRecHitsPreSplittingSoA, cuda = _siPixelRecHitSoAFromCUDAPhase2.clone())

# transfer the pixel rechits to the host and convert them from SoA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDAPhase1_cfi import siPixelRecHitFromCUDAPhase1 as _siPixelRecHitFromCUDA
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDAHIonPhase1_cfi import siPixelRecHitFromCUDAHIonPhase1 as _siPixelRecHitFromCUDAHIonPhase1
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromCUDAPhase2_cfi import siPixelRecHitFromCUDAPhase2 as _siPixelRecHitFromCUDAPhase2

(gpu & pixelNtupletFit).toModify(siPixelRecHitsPreSplitting, cuda = _siPixelRecHitFromCUDA.clone())
(gpu & pixelNtupletFit & pp_on_AA).toModify(siPixelRecHitsPreSplitting, cuda = _siPixelRecHitFromCUDAHIonPhase1.clone())
(gpu & pixelNtupletFit & phase2_tracker).toModify(siPixelRecHitsPreSplitting, cuda = _siPixelRecHitFromCUDAPhase2.clone())


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


#(gpu & pixelNtupletFit & phase2_tracker).toReplaceWith(siPixelRecHitsPreSplitting , cuda = _siPixelRecHitFromCUDAPhase2.clone())

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
