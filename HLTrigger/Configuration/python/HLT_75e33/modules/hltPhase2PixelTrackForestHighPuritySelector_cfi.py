import FWCore.ParameterSet.Config as cms

# Final selection of the pixel tracks of the vector hit CA: a gradient-boosted forest on 31 features
# (17 fit and covariance features, 10 hit and vector hit features, 4 hit-walk features).
hltPhase2PixelTrackForestHighPuritySelector = cms.EDProducer('PixelTrackForestHighPuritySelector@alpaka',
    pixelTrackSrc = cms.InputTag('hltPhase2PixelTracksSoA'),
    maxNumberOfTracks = cms.int32(2*60*1024),
    maxPreselectedTracks = cms.int32(9_984),     # sized for PU200 with headroom, so nothing is truncated
    minNumberOfHits = cms.int32(0),
    avgHitsPerTrack = cms.int32(8),
    minimumTrackQuality = cms.string('tight'),
    # Compact gradient-boosted forest binary, shared per device.
    model = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/itaca_selectors/prompt_tree31_wp_20261117.bin'),
    scoreThreshold = cms.double(0.520),
    # scoreThresholdLowDxy < 0 disables the dxy-dependent threshold ramp: one flat score cut.
    scoreThresholdLowDxy = cms.double(-1.0),
    dxyRampKnee = cms.double(2.0),
    # The hit features read the pixel rechits and vector hits the CA indexed, and the raw OT rechits.
    useHitFeatures = cms.bool(True),
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    vectorHitsSrc = cms.InputTag('hltPhase2OTVectorHitsSoA'),
    otRecHitsSoASrc = cms.InputTag('hltCAOTRecHitsSoA')
)
