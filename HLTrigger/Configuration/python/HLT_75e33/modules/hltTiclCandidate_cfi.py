import FWCore.ParameterSet.Config as cms

hltTiclCandidate = cms.EDProducer("TICLCandidateProducer",
    cutTk = cms.string('1.48 < abs(eta) < 3.0 && pt > 1. && quality("highPurity") && hitPattern().numberOfLostHits("MISSING_OUTER_HITS") < 5'),
    detector = cms.string('HGCAL'),
    egamma_tracksterlinks_collections = cms.VInputTag("hltTiclTracksterLinks"),
    egamma_tracksters_collections = cms.VInputTag("hltTiclTracksterLinks"),
    general_tracksterlinks_collections = cms.VInputTag("hltTiclTracksterLinks"),
    general_tracksters_collections = cms.VInputTag("hltTiclTracksterLinks"),
    interpretationDescPSet = cms.PSet(
        algo_verbosity = cms.int32(0),
        cutTk = cms.string('1.48 < abs(eta) < 3.0 && pt > 1. && quality("highPurity") && hitPattern().numberOfLostHits("MISSING_OUTER_HITS") < 5'),
        delta_tk_ts_interface = cms.double(0.03),
        delta_tk_ts_layer1 = cms.double(0.02),
        timing_quality_threshold = cms.double(0.5),
        type = cms.string('General')
    ),
    layer_clusters = cms.InputTag("hgcalMergeLayerClusters"),
    layer_clustersTime = cms.InputTag("hgcalMergeLayerClusters","timeLayerCluster"),
    mightGet = cms.optional.untracked.vstring,
    muons = cms.InputTag("hltPhase2L3Muons"),
    original_masks = cms.VInputTag("hgcalMergeLayerClusters:InitialLayerClustersMask"),
    propagator = cms.string('PropagatorWithMaterial'),
    timingQualityThreshold = cms.double(0.5),
    timingSoA = cms.InputTag("mtdSoA"),
    tracks = cms.InputTag("generalTracks"),
    useMTDTiming = cms.bool(False),
    useTimingAverage = cms.bool(False)
)

