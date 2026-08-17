import FWCore.ParameterSet.Config as cms

hltPhase2PixelTrackTorchHighPuritySelector = cms.EDProducer('PixelTrackTorchHighPuritySelector@alpaka',
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
    maxNumberOfTracks = cms.int32(2*60*1024),
    maxPreselectedTracks = cms.int32(9_984),
    minNumberOfHits = cms.int32(0),
    avgHitsPerTrack = cms.int32(8),
    minimumTrackQuality = cms.string('tight'),
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/pixel_track_classifier_FP16.pt'),
    scoreThreshold = cms.double(0.4),
    batchSize = cms.int32(4_992)
)

# Vector hit chain (phase2CAVectorHits): 31-feature gradient-boosted forest (17 fit/covariance, 10 hit/vector hit CA,
# 4 hit-walk features). toReplaceWith keeps the module label, so downstream wiring is untouched.
_hltPhase2PixelTrackForestHighPuritySelector = cms.EDProducer('PixelTrackForestHighPuritySelector@alpaka',
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
    maxNumberOfTracks = cms.int32(2*60*1024),
    maxPreselectedTracks = cms.int32(9_984),     # sized for PU200 with headroom, so nothing is truncated
    minNumberOfHits = cms.int32(0),
    avgHitsPerTrack = cms.int32(8),
    minimumTrackQuality = cms.string('tight'),
    # Compact gradient-boosted forest binary, shared per device.
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/prompt_tree31_wp_20260914.bin'),
    scoreThreshold = cms.double(0.232),
    # scoreThresholdLowDxy < 0 disables the dxy-dependent threshold ramp: one flat score cut.
    scoreThresholdLowDxy = cms.double(-1.0),
    dxyRampKnee = cms.double(2.0),
    # The hit features read the pixel rechits and vector hits the CA indexed, and the raw OT rechits.
    useHitFeatures = cms.bool(True),
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    vectorHitsSrc = cms.InputTag('hltPhase2OTVectorHitsSoA'),
    otRecHitsSoASrc = cms.InputTag('hltPixelSeedingOTRecHitsSoA')
)

from Configuration.ProcessModifiers.phase2CAVectorHits_cff import phase2CAVectorHits
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
(phase2CAVectorHits | ngtScouting).toReplaceWith(
        hltPhase2PixelTrackTorchHighPuritySelector,
        _hltPhase2PixelTrackForestHighPuritySelector
)
