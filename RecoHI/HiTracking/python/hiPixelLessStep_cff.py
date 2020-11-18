from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
from .HIPixelTripletSeeds_cff import *
from .HIPixel3PrimTracks_cfi import *

##########################################################################
# Large impact parameter tracking using TIB/TID/TEC stereo layer seeding #
##########################################################################

#HIClusterRemover
from RecoHI.HiTracking.hiMixedTripletStep_cff import hiMixedTripletStepClusters
hiPixelLessStepClusters = hiMixedTripletStepClusters.clone(
    trajectories     = "hiMixedTripletStepTracks",
    overrideTrkQuals = 'hiMixedTripletStepSelector:hiMixedTripletStep'
)
# SEEDING LAYERS
pixelLessStepSeedLayers.TIB.skipClusters   = 'hiPixelLessStepClusters'
pixelLessStepSeedLayers.MTIB.skipClusters   = 'hiPixelLessStepClusters'
pixelLessStepSeedLayers.TID.skipClusters   = 'hiPixelLessStepClusters'
pixelLessStepSeedLayers.MTID.skipClusters   = 'hiPixelLessStepClusters'
pixelLessStepSeedLayers.TEC.skipClusters   = 'hiPixelLessStepClusters'
pixelLessStepSeedLayers.MTEC.skipClusters   = 'hiPixelLessStepClusters'

# TrackingRegion
from RecoHI.HiTracking.hiMixedTripletStep_cff import hiMixedTripletStepTrackingRegionsA as _hiMixedTripletStepTrackingRegionsA
hiPixelLessStepTrackingRegions = _hiMixedTripletStepTrackingRegionsA.clone(RegionPSet=dict(
     fixedError = 3.0,#12.0
     ptMin = 0.7, #0.4
     originRadius = 1.0,
     maxPtMin = 1.0,#0.7
))

# seeding
pixelLessStepHitDoublets.clusterCheck = ""
pixelLessStepHitDoublets.trackingRegions = "hiPixelLessStepTrackingRegions"

# QUALITY CUTS DURING TRACK BUILDING
from RecoTracker.IterativeTracking.PixelLessStep_cff import pixelLessStepTrajectoryFilter
pixelLessStepTrajectoryFilter.minimumNumberOfHits = 5
pixelLessStepTrajectoryFilter.minPt = 0.7

# MAKING OF TRACK CANDIDATES
pixelLessStepTrackCandidates.clustersToSkip = 'hiPixelLessStepClusters'

# TRACK FITTING
hiPixelLessStepTracks = pixelLessStepTracks.clone()

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiPixelLessStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src = 'hiPixelLessStepTracks',
    useAnyMVA = False,
    GBRForestLabel = 'HIMVASelectorIter12',
    GBRForestVars = ['chi2perdofperlayer', 'nhits', 'nlayers', 'eta'],
    trackSelectors= cms.VPSet(
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
           name = 'hiPixelLessStepLoose',
           applyAdaptedPVCuts = False,
           useMVA = False,
       ), #end of pset
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
           name = 'hiPixelLessStepTight',
           preFilterName = 'hiPixelLessStepLoose',
           applyAdaptedPVCuts = False,
           useMVA = False,
           minMVA = -0.2
       ),
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
           name = 'hiPixelLessStep',
           preFilterName = 'hiPixelLessStepTight',
           applyAdaptedPVCuts = False,
           useMVA = False,
           minMVA = -0.09
       ),
    ) #end of vpset
) #end of clone

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiPixelLessStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ['hiPixelLessStepTracks'],
    hasSelector = [1],
    selectedTrackQuals = ["hiPixelLessStepSelector:hiPixelLessStep"],
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
)

hiPixelLessStepTask = cms.Task(hiPixelLessStepClusters,
                             pixelLessStepSeedLayers,
                             hiPixelLessStepTrackingRegions,
                             pixelLessStepHitDoublets,
                             pixelLessStepHitTriplets,
                             pixelLessStepSeeds,
                             pixelLessStepTrackCandidates,
                             hiPixelLessStepTracks,
                             hiPixelLessStepSelector,
                             hiPixelLessStepQual
                             )
hiPixelLessStep = cms.Sequence(hiPixelLessStepTask)
