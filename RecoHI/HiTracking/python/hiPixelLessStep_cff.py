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
hiPixelLessStepClusters = hiMixedTripletStepClusters.clone()
hiPixelLessStepClusters.trajectories = cms.InputTag("hiMixedTripletStepTracks")
hiPixelLessStepClusters.overrideTrkQuals = cms.InputTag('hiMixedTripletStepSelector','hiMixedTripletStep')

# SEEDING LAYERS
pixelLessStepSeedLayers.TIB.skipClusters   = cms.InputTag('hiPixelLessStepClusters')
pixelLessStepSeedLayers.MTIB.skipClusters   = cms.InputTag('hiPixelLessStepClusters')
pixelLessStepSeedLayers.TID.skipClusters   = cms.InputTag('hiPixelLessStepClusters')
pixelLessStepSeedLayers.MTID.skipClusters   = cms.InputTag('hiPixelLessStepClusters')
pixelLessStepSeedLayers.TEC.skipClusters   = cms.InputTag('hiPixelLessStepClusters')
pixelLessStepSeedLayers.MTEC.skipClusters   = cms.InputTag('hiPixelLessStepClusters')

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
pixelLessStepTrackCandidates.clustersToSkip = cms.InputTag('hiPixelLessStepClusters')

# TRACK FITTING
hiPixelLessStepTracks = pixelLessStepTracks.clone()

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiPixelLessStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiPixelLessStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('HIMVASelectorIter12'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiPixelLessStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiPixelLessStepTight',
    preFilterName = 'hiPixelLessStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.2)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiPixelLessStep',
    preFilterName = 'hiPixelLessStepTight',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.09)
    ),
    ) #end of vpset
    ) #end of clone

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiPixelLessStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers=cms.VInputTag(cms.InputTag('hiPixelLessStepTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiPixelLessStepSelector","hiPixelLessStep")),
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
