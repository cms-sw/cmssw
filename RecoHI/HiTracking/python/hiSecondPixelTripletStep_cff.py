import FWCore.ParameterSet.Config as cms

# Filter on quality tracks
hiFirstStepFilter = cms.EDProducer("QualityFilter",
                                 TrackQuality = cms.string('highPurity'),
                                 recTracks = cms.InputTag("hiSelectedTracks")
                                 )

# NEW CLUSTERS (remove previously used clusters)
hiSecondPixelTripletClusters = cms.EDProducer("TrackClusterRemover",
                                clusterLessSolution= cms.bool(True),
                                trajectories = cms.InputTag("hiFirstStepFilter"),
                                TrackQuality = cms.string('highPurity'),
                                pixelClusters = cms.InputTag("siPixelClusters"),
                                stripClusters = cms.InputTag("siStripClusters"),
                                Common = cms.PSet(
    maxChi2 = cms.double(9.0)
    ),
                                Strip = cms.PSet(
    #Yen-Jie's mod to preserve merged clusters
    maxSize = cms.uint32(2),
    maxChi2 = cms.double(9.0)
    )
                                )


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
hiSecondPixelTripletSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
hiSecondPixelTripletSeedLayers.BPix.skipClusters = cms.InputTag('hiSecondPixelTripletClusters')
hiSecondPixelTripletSeedLayers.FPix.skipClusters = cms.InputTag('hiSecondPixelTripletClusters')

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
hiSecondPixelTripletSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 4.0,
    originRadius = 0.005,
    nSigmaZ = 4.0
    )
    )
)

hiSecondPixelTripletSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'hiSecondPixelTripletSeedLayers'
hiSecondPixelTripletSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = 5000000
hiSecondPixelTripletSeeds.ClusterCheckPSet.MaxNumberOfPixelClusters = 5000000
hiSecondPixelTripletSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 50000000

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
hiSecondPixelTripletSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiSecondPixelTripletTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    maxLostHits = 1,
    minimumNumberOfHits = 6,
    minPt = 1.0
    )

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
hiSecondPixelTripletChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
        ComponentName = cms.string('hiSecondPixelTripletChi2Est'),
            nSigma = cms.double(3.0),
            MaxChi2 = cms.double(9.0)
        )


# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiSecondPixelTripletTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('hiSecondPixelTripletTrajectoryFilter')),
    clustersToSkip = cms.InputTag('hiSecondPixelTripletClusters'),
    maxCand = 3,
    #estimator = cms.string('hiSecondPixelTripletChi2Est')
    )


# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiSecondPixelTripletTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('hiSecondPixelTripletSeeds'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiSecondPixelTripletTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiSecondPixelTripletGlobalPrimTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'hiSecondPixelTripletTrackCandidates',
    AlgorithmName = cms.string('iter1')
    )



# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiSecondPixelTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiSecondPixelTripletGlobalPrimTracks',
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiSecondPixelTripletStepLoose',
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiSecondPixelTripletStepTight',
    preFilterName = 'hiSecondPixelTripletStepLoose',
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiSecondPixelTripletStep',
    preFilterName = 'hiSecondPixelTripletStepTight',
    min_nhits = 14
    ),
    ) #end of vpset
    ) #end of clone


import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiSecondQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = cms.VInputTag(cms.InputTag('hiSecondPixelTripletGlobalPrimTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiSecondPixelTripletStepSelector","hiSecondPixelTripletStep")),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
    #writeOnlyTrkQuals = True
    )

# Final sequence

hiSecondPixelTripletStep = cms.Sequence(hiFirstStepFilter*
                                        hiSecondPixelTripletClusters*
                                        hiSecondPixelTripletSeedLayers*
                                        hiSecondPixelTripletSeeds*
                                        hiSecondPixelTripletTrackCandidates*
                                        hiSecondPixelTripletGlobalPrimTracks*
                                        hiSecondPixelTripletStepSelector
                                        *hiSecondQual
                                        )
