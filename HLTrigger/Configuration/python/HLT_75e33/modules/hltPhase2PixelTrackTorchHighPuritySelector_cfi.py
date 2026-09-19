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

# Stub chain (phase2CAStubs): 31-feature gradient-boosted forest (17 fit/covariance, 10 hit/stub CA,
# 4 hit-walk features). toReplaceWith keeps the module label, so downstream wiring is untouched.
_hltPhase2PixelTrackForestHighPuritySelector = cms.EDProducer('PixelTrackForestHighPuritySelector@alpaka',
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
    maxNumberOfTracks = cms.int32(2*60*1024),
    maxPreselectedTracks = cms.int32(9_984),     # sized for PU200 with headroom, so nothing is truncated
    # Compact gradient-boosted forest binary, shared per device.
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/prompt_tree31_wp_20260914.bin'),
    # scoreThresholdLowDxy < 0 disables the dxy-dependent threshold ramp: one flat score cut.
    scoreThreshold = cms.double(0.232)
)

from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
phase2CAStubs.toReplaceWith(hltPhase2PixelTrackTorchHighPuritySelector,
                            _hltPhase2PixelTrackForestHighPuritySelector)
