from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *
from .HIPixelTripletSeeds_cff import *
from .HIPixel3PrimTracks_cfi import *

#cluster remover
hiMixedTripletStepClusters = cms.EDProducer("HITrackClusterRemover",
     clusterLessSolution = cms.bool(True),
     trajectories = cms.InputTag("hiPixelPairGlobalPrimTracks"),
     overrideTrkQuals = cms.InputTag('hiPixelPairStepSelector','hiPixelPairStep'),
     TrackQuality = cms.string('highPurity'),
     minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
     pixelClusters = cms.InputTag("siPixelClusters"),
     stripClusters = cms.InputTag("siStripClusters"),
     Common = cms.PSet(
         maxChi2 = cms.double(9.0),
     ),
     Strip = cms.PSet(
        #Yen-Jie's mod to preserve merged clusters
        maxSize = cms.uint32(2),
        maxChi2 = cms.double(9.0)
     )
)

# SEEDING LAYERS
mixedTripletStepSeedLayersA.layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 
                                        'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg', 
                                        'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg')
mixedTripletStepSeedLayersA.BPix.skipClusters = cms.InputTag('hiMixedTripletStepClusters')
mixedTripletStepSeedLayersA.FPix.skipClusters = cms.InputTag('hiMixedTripletStepClusters')
mixedTripletStepSeedLayersA.TEC.skipClusters = cms.InputTag('hiMixedTripletStepClusters')

# TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
hiMixedTripletStepTrackingRegionsA = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
     VertexCollection = "hiSelectedPixelVertex",
     fixedError = 3.75,#15.0
     ptMin = 0.4,
     originRadius = 1.5,
     originRScaling4BigEvts = cms.bool(True),
     halfLengthScaling4BigEvts = cms.bool(False),
     ptMinScaling4BigEvts = cms.bool(True),
     minOriginR = 0,
     minHalfLength = 0,
     maxPtMin = 0.7,
     scalingStartNPix = 20000,
     scalingEndNPix = 35000     
))

# seeding
mixedTripletStepHitDoubletsA.clusterCheck = ""
mixedTripletStepHitDoubletsA.trackingRegions = "hiMixedTripletStepTrackingRegionsA"

# SEEDING LAYERS
mixedTripletStepSeedLayersB.BPix.skipClusters = cms.InputTag('hiMixedTripletStepClusters')
mixedTripletStepSeedLayersB.TIB.skipClusters = cms.InputTag('hiMixedTripletStepClusters')

hiMixedTripletStepTrackingRegionsB = hiMixedTripletStepTrackingRegionsA.clone(RegionPSet=dict(
     originRadius = 1.0,
))


# seeding
mixedTripletStepHitDoubletsB.clusterCheck = ""
mixedTripletStepHitDoubletsB.trackingRegions = "hiMixedTripletStepTrackingRegionsB"

# QUALITY CUTS DURING TRACK BUILDING
from RecoTracker.IterativeTracking.MixedTripletStep_cff import mixedTripletStepTrajectoryFilter
mixedTripletStepTrajectoryFilter.minimumNumberOfHits = 5
mixedTripletStepTrajectoryFilter.minPt = 0.4
mixedTripletStepPropagator.ptMin = 0.4
mixedTripletStepPropagatorOpposite.ptMin = 0.4

# MAKING OF TRACK CANDIDATES
mixedTripletStepTrackCandidates.clustersToSkip = cms.InputTag('hiMixedTripletStepClusters')

# TRACK FITTING
hiMixedTripletStepTracks = mixedTripletStepTracks.clone()

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiMixedTripletStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiMixedTripletStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('HIMVASelectorIter11'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiMixedTripletStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiMixedTripletStepTight',
    preFilterName = 'hiMixedTripletStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.2)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiMixedTripletStep',
    preFilterName = 'hiMixedTripletStepTight',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.09)
    ),
    ) #end of vpset
    ) #end of clone

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiMixedTripletStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers=cms.VInputTag(cms.InputTag('hiMixedTripletStepTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiMixedTripletStepSelector","hiMixedTripletStep")),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
    )

hiMixedTripletStepTask = cms.Task(
                                hiMixedTripletStepClusters,
                                mixedTripletStepSeedLayersA,
                                hiMixedTripletStepTrackingRegionsA,
                                mixedTripletStepHitDoubletsA,
                                mixedTripletStepHitTripletsA,
                                mixedTripletStepSeedsA,
                                mixedTripletStepSeedLayersB,
                                hiMixedTripletStepTrackingRegionsB,
                                mixedTripletStepHitDoubletsB,
                                mixedTripletStepHitTripletsB,
                                mixedTripletStepSeedsB,
                                mixedTripletStepSeeds,
                                mixedTripletStepTrackCandidates,
                                hiMixedTripletStepTracks,
                                hiMixedTripletStepSelector,
                                hiMixedTripletStepQual)
hiMixedTripletStep = cms.Sequence(hiMixedTripletStepTask)
