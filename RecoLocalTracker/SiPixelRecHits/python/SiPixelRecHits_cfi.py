import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *
from Configuration.ProcessModifiers.alpaka_cff import alpaka


# legacy pixel rechit producer
from RecoLocalTracker.SiPixelRecHits.siPixelRecHitConverter_cfi import siPixelRecHitConverter as _siPixelRecHitConverter
siPixelRecHits = _siPixelRecHitConverter.clone()

# HIon Modifiers
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
# Phase 2 Tracker Modifier
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker

# The legacy pixel rechit producer
siPixelRecHitsPreSplitting = siPixelRecHits.clone(
        src = 'siPixelClustersPreSplitting'
)

siPixelRecHitsPreSplittingTask = cms.Task(
    siPixelRecHitsPreSplitting
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
            src = cms.InputTag('siPixelClustersPreSplitting'))
)

(alpaka & phase2_tracker).toModify(siPixelRecHitsPreSplitting,
    cpu = _siPixelRecHitFromSoAAlpakaPhase2.clone(
            pixelRecHitSrc = cms.InputTag('siPixelRecHitsPreSplittingAlpaka'),
            src = cms.InputTag('siPixelClustersPreSplitting'))
)

(alpaka & pp_on_AA & ~phase2_tracker).toModify(siPixelRecHitsPreSplitting,
    cpu = _siPixelRecHitFromSoAAlpakaHIonPhase1.clone(
            pixelRecHitSrc = cms.InputTag('siPixelRecHitsPreSplittingAlpaka'),
            src = cms.InputTag('siPixelClustersPreSplitting'))
)


alpaka.toReplaceWith(siPixelRecHitsPreSplittingTask, cms.Task(
                        # Reconstruct the pixel hits with alpaka on the device
                        siPixelRecHitsPreSplittingAlpaka,
                        # Reconstruct the pixel hits with alpaka on the cpu (if requested by the validation)
                        siPixelRecHitsPreSplittingAlpakaSerial,
                        # Convert hit soa on host to legacy formats
                        siPixelRecHitsPreSplitting))
