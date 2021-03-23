from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
from RecoTracker.IterativeTracking.TobTecStep_cff import *
from .HIPixelTripletSeeds_cff import *
from .HIPixel3PrimTracks_cfi import *

#######################################################################
# Very large impact parameter tracking using TOB + TEC ring 5 seeding #
#######################################################################
from RecoHI.HiTracking.hiPixelLessStep_cff import hiPixelLessStepClusters
hiTobTecStepClusters = hiPixelLessStepClusters.clone(
    trajectories = "hiPixelLessStepTracks",
    overrideTrkQuals = 'hiPixelLessStepSelector:hiPixelLessStep'
)
# TRIPLET SEEDING LAYERS
tobTecStepSeedLayersTripl.TOB.skipClusters   = 'hiTobTecStepClusters'
tobTecStepSeedLayersTripl.MTOB.skipClusters   = 'hiTobTecStepClusters'
tobTecStepSeedLayersTripl.MTEC.skipClusters   = 'hiTobTecStepClusters'

# Triplet TrackingRegion
from RecoHI.HiTracking.hiMixedTripletStep_cff import hiMixedTripletStepTrackingRegionsA as _hiMixedTripletStepTrackingRegionsA
hiTobTecStepTrackingRegionsTripl = _hiMixedTripletStepTrackingRegionsA.clone(RegionPSet=dict(
     fixedError = 5.0,#20.0
     ptMin = 0.9,#0.55
     originRadius = 3.5,
     maxPtMin = 1.2,#0.85
))

# Triplet seeding
tobTecStepHitDoubletsTripl.clusterCheck = ""
tobTecStepHitDoubletsTripl.trackingRegions = "hiTobTecStepTrackingRegionsTripl"

tobTecStepSeedLayersPair.TOB.skipClusters   = 'hiTobTecStepClusters'
tobTecStepSeedLayersPair.TEC.skipClusters   = 'hiTobTecStepClusters'

# Pair TrackingRegion
hiTobTecStepTrackingRegionsPair = hiTobTecStepTrackingRegionsTripl.clone(RegionPSet=dict(
     fixedError = 7.5,#30.0
     originRadius = 6.0,
     maxPtMin = 1.5,#0.9
))

# Pair seeds
tobTecStepHitDoubletsPair.clusterCheck = ""
tobTecStepHitDoubletsPair.trackingRegions = "hiTobTecStepTrackingRegionsPair"


# QUALITY CUTS DURING TRACK BUILDING (for inwardss and outwards track building steps)
from RecoTracker.IterativeTracking.TobTecStep_cff import tobTecStepTrajectoryFilter
tobTecStepTrajectoryFilter.minimumNumberOfHits = 5
tobTecStepTrajectoryFilter.minPt = 0.85

# MAKING OF TRACK CANDIDATES
tobTecStepTrackCandidates.clustersToSkip = 'hiTobTecStepClusters'

# TRACK FITTING
hiTobTecStepTracks = tobTecStepTracks.clone()

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiTobTecStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src = 'hiTobTecStepTracks',
    useAnyMVA = False,
    GBRForestLabel = 'HIMVASelectorIter13',
    GBRForestVars = ['chi2perdofperlayer', 'nhits', 'nlayers', 'eta'],
    trackSelectors = cms.VPSet(
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
           name = 'hiTobTecStepLoose',
           applyAdaptedPVCuts = False,
           useMVA = False,
       ), #end of pset
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
           name = 'hiTobTecStepTight',
           preFilterName = 'hiTobTecStepLoose',
           applyAdaptedPVCuts = False,
           useMVA = False,
           minMVA = -0.2
       ),
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
           name = 'hiTobTecStep',
           preFilterName = 'hiTobTecStepTight',
           applyAdaptedPVCuts = False,
           useMVA = False,
           minMVA = -0.09
       ),
    ) #end of vpset
) #end of clone

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiTobTecStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = ['hiTobTecStepTracks'],
    hasSelector = [1],
    selectedTrackQuals = ["hiTobTecStepSelector:hiTobTecStep"],
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
)


hiTobTecStepTask = cms.Task(hiTobTecStepClusters,
                          tobTecStepSeedLayersTripl,
                          hiTobTecStepTrackingRegionsTripl,
                          tobTecStepHitDoubletsTripl,
                          tobTecStepHitTripletsTripl,
                          tobTecStepSeedsTripl,
                          tobTecStepSeedLayersPair,
                          hiTobTecStepTrackingRegionsPair,
                          tobTecStepHitDoubletsPair,
                          tobTecStepSeedsPair,
                          tobTecStepSeeds,
                          tobTecStepTrackCandidates,
                          hiTobTecStepTracks,
                          hiTobTecStepSelector,
                          hiTobTecStepQual
                          )
hiTobTecStep = cms.Sequence(hiTobTecStepTask)
