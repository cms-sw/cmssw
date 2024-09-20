import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from Configuration.ProcessModifiers.alpaka_cff import alpaka
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

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

# convert the pixel rechits from legacy to SoA format on CPU
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyPhase1_cfi import siPixelRecHitSoAFromLegacyPhase1 as _siPixelRecHitsPreSplittingSoA
siPixelRecHitsPreSplittingCPU = _siPixelRecHitsPreSplittingSoA.clone(
    convertToLegacy = True
)

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyHIonPhase1_cfi import siPixelRecHitSoAFromLegacyHIonPhase1 as _siPixelRecHitsPreSplittingSoAHIonPhase1
(pp_on_AA & ~phase2_tracker).toReplaceWith(siPixelRecHitsPreSplittingCPU,
    _siPixelRecHitsPreSplittingSoAHIonPhase1.clone(
        convertToLegacy = True,
        CPE = cms.string('PixelCPEFastHIonPhase1')
    )
)

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitSoAFromLegacyPhase2_cfi import siPixelRecHitSoAFromLegacyPhase2 as _siPixelRecHitsPreSplittingSoAPhase2
phase2_tracker.toReplaceWith(siPixelRecHitsPreSplittingCPU,
    _siPixelRecHitsPreSplittingSoAPhase2.clone(
        convertToLegacy = True,
        CPE = cms.string('PixelCPEFastPhase2')
    )
)

# modifier used to prompt patatrack pixel tracks reconstruction on cpu
from Configuration.ProcessModifiers.pixelNtupletFit_cff import pixelNtupletFit
pixelNtupletFit.toModify(siPixelRecHitsPreSplitting,
    cpu = cms.EDAlias(
        siPixelRecHitsPreSplittingCPU = cms.VPSet(
            cms.PSet(type = cms.string("SiPixelRecHitedmNewDetSetVector")),
            cms.PSet(type = cms.string("uintAsHostProduct"))
        )
    )
)

siPixelRecHitsPreSplittingTask = cms.Task(
    # SwitchProducer wrapping the legacy pixel rechit producer or the cpu SoA producer
    siPixelRecHitsPreSplitting
)

# this is an alias for the SoA on CPU to be used for DQM
siPixelRecHitsPreSplittingSoA = SwitchProducerCUDA(
    cpu = cms.EDAlias(
        siPixelRecHitsPreSplittingCPU = cms.VPSet(
            cms.PSet(type = cms.string("pixelTopologyPhase1TrackingRecHitSoAHost")),
            cms.PSet(type = cms.string("uintAsHostProduct"))
        )
    )
)

(pp_on_AA & ~phase2_tracker).toModify(siPixelRecHitsPreSplittingSoA,
    cpu = cms.EDAlias(
        siPixelRecHitsPreSplittingCPU = cms.VPSet(
             cms.PSet(type = cms.string("pixelTopologyHIonPhase1TrackingRecHitSoAHost")),
             cms.PSet(type = cms.string("uintAsHostProduct"))
         )
    )
)

phase2_tracker.toModify(siPixelRecHitsPreSplittingSoA,
    cpu = cms.EDAlias(
        siPixelRecHitsPreSplittingCPU = cms.VPSet(
             cms.PSet(type = cms.string("pixelTopologyPhase2TrackingRecHitSoAHost")),
             cms.PSet(type = cms.string("uintAsHostProduct"))
         )
    )
)

pixelNtupletFit.toReplaceWith(siPixelRecHitsPreSplittingTask,
    cms.Task(
        # reconstruct the pixel rechits on the cpu
        siPixelRecHitsPreSplittingCPU,
        # SwitchProducer wrapping an EDAlias on cpu
        siPixelRecHitsPreSplittingTask.copy(),
        # producing and converting on cpu (if needed)
        siPixelRecHitsPreSplittingSoA
    )
)

######################################################################

### Alpaka Pixel Hits Reco
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitAlpakaPhase1_cfi import siPixelRecHitAlpakaPhase1 as _siPixelRecHitAlpakaPhase1
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitAlpakaPhase2_cfi import siPixelRecHitAlpakaPhase2 as _siPixelRecHitAlpakaPhase2
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitAlpakaHIonPhase1_cfi import siPixelRecHitAlpakaHIonPhase1 as _siPixelRecHitAlpakaHIonPhase1


# Hit SoA producer on the device
siPixelRecHitsPreSplittingAlpaka = _siPixelRecHitAlpakaPhase1.clone(
    src = "siPixelClustersPreSplittingAlpaka"
)
phase2_tracker.toReplaceWith(siPixelRecHitsPreSplittingAlpaka,_siPixelRecHitAlpakaPhase2.clone(
    src = "siPixelClustersPreSplittingAlpaka"
))
(pp_on_AA & ~phase2_tracker).toReplaceWith(siPixelRecHitsPreSplittingAlpaka,_siPixelRecHitAlpakaHIonPhase1.clone(
    src = "siPixelClustersPreSplittingAlpaka"
))

# Hit SoA producer on the cpu, for validation
siPixelRecHitsPreSplittingAlpakaSerial = makeSerialClone(siPixelRecHitsPreSplittingAlpaka,
    src = "siPixelClustersPreSplittingAlpakaSerial"
)

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromSoAAlpakaPhase1_cfi import siPixelRecHitFromSoAAlpakaPhase1 as _siPixelRecHitFromSoAAlpakaPhase1
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromSoAAlpakaPhase2_cfi import siPixelRecHitFromSoAAlpakaPhase2 as _siPixelRecHitFromSoAAlpakaPhase2
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromSoAAlpakaHIonPhase1_cfi import siPixelRecHitFromSoAAlpakaHIonPhase1 as _siPixelRecHitFromSoAAlpakaHIonPhase1

(alpaka & ~phase2_tracker).toModify(siPixelRecHitsPreSplitting,
    cpu = _siPixelRecHitFromSoAAlpakaPhase1.clone(
        pixelRecHitSrc = cms.InputTag('siPixelRecHitsPreSplittingAlpaka'),
        src = cms.InputTag('siPixelClustersPreSplitting')
    )
)

(alpaka & phase2_tracker).toModify(siPixelRecHitsPreSplitting,
    cpu = _siPixelRecHitFromSoAAlpakaPhase2.clone(
        pixelRecHitSrc = cms.InputTag('siPixelRecHitsPreSplittingAlpaka'),
        src = cms.InputTag('siPixelClustersPreSplitting')
    )
)

(alpaka & pp_on_AA & ~phase2_tracker).toModify(siPixelRecHitsPreSplitting,
    cpu = _siPixelRecHitFromSoAAlpakaHIonPhase1.clone(
        pixelRecHitSrc = cms.InputTag('siPixelRecHitsPreSplittingAlpaka'),
        src = cms.InputTag('siPixelClustersPreSplitting')
    )
)


alpaka.toReplaceWith(siPixelRecHitsPreSplittingTask, cms.Task(
                        # Reconstruct the pixel hits with alpaka on the device
                        siPixelRecHitsPreSplittingAlpaka,
                        # Reconstruct the pixel hits with alpaka on the cpu (if requested by the validation)
                        siPixelRecHitsPreSplittingAlpakaSerial,
                        # Convert hit soa on host to legacy formats
                        siPixelRecHitsPreSplitting))
