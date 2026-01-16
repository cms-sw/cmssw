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

### Phase-2 CA OT extension
from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension

from RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi import siPhase2RecHits

from RecoLocalTracker.Phase2TrackerRecHits.phase2OTRecHitsSoAConverter_cfi import phase2OTRecHitsSoAConverter as _phase2OTRecHitsSoAConverter
phase2OTRecHitsSoAConverter = _phase2OTRecHitsSoAConverter.clone(
    beamSpot = "offlineBeamSpot",
    otRecHitSource = "siPhase2RecHits",
    pixelRecHitSoASource = "siPixelRecHitsPreSplittingAlpaka"
)

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitExtendedAlpaka_cfi import siPixelRecHitExtendedAlpaka as _siPixelRecHitExtendedAlpaka
siPixelRecHitsExtendedPreSplittingAlpaka = _siPixelRecHitExtendedAlpaka.clone(
    pixelRecHitsSoA = "siPixelRecHitsPreSplittingAlpaka",
    trackerRecHitsSoA = "phase2OTRecHitsSoAConverter"
)

# Hit SoA producer on the cpu, for validation
siPixelRecHitsPreSplittingAlpakaSerial = makeSerialClone(siPixelRecHitsPreSplittingAlpaka,
                                                         src = "siPixelClustersPreSplittingAlpakaSerial")

siPixelRecHitsExtendedPreSplittingAlpakaSerial = makeSerialClone(siPixelRecHitsExtendedPreSplittingAlpaka,
                                                                pixelRecHitsSoA = "siPixelClustersPreSplittingAlpakaSerial")

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitFromSoAAlpaka_cfi import siPixelRecHitFromSoAAlpaka as _siPixelRecHitFromSoAAlpaka

alpaka.toReplaceWith(siPixelRecHitsPreSplitting, _siPixelRecHitFromSoAAlpaka.clone(
            pixelRecHitSrc = cms.InputTag('siPixelRecHitsPreSplittingAlpaka'),
            src = cms.InputTag('siPixelClustersPreSplitting'))
)

(alpaka & pp_on_AA & ~phase2_tracker).toModify(siPixelRecHitsPreSplitting,
            maxHitsInModules = cms.uint32(2048)
)


(~phase2CAExtension & alpaka).toReplaceWith(siPixelRecHitsPreSplittingTask, cms.Task(
                        # Reconstruct the pixel hits with alpaka on the device
                        siPixelRecHitsPreSplittingAlpaka,
                        # Reconstruct the pixel hits with alpaka on the cpu (if requested by the validation)
                        siPixelRecHitsPreSplittingAlpakaSerial,
                        # Convert hit soa on host to legacy formats
                        siPixelRecHitsPreSplitting))

phase2CAExtension.toReplaceWith(siPixelRecHitsPreSplittingTask, cms.Task(
    siPhase2RecHits,
    siPixelRecHitsPreSplittingAlpaka,
    siPixelRecHitsPreSplittingAlpakaSerial,    
    phase2OTRecHitsSoAConverter,
    siPixelRecHitsExtendedPreSplittingAlpaka,
    siPixelRecHitsExtendedPreSplittingAlpakaSerial,
    siPixelRecHitsPreSplitting
))
