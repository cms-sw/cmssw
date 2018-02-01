import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg
from RecoTracker.IterativeTracking.TobTecStep_cff import *
from HIPixelTripletSeeds_cff import *
from HIPixel3PrimTracks_cfi import *

#######################################################################
# Very large impact parameter tracking using TOB + TEC ring 5 seeding #
#######################################################################
hiTobTecStepClusters = cms.EDProducer("HITrackClusterRemover",
     clusterLessSolution = cms.bool(True),
     trajectories = cms.InputTag("hiPixelLessStepTracks"),
     overrideTrkQuals = cms.InputTag('hiPixelLessStepSelector','hiPixelLessStep'),
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

# TRIPLET SEEDING LAYERS
tobTecStepSeedLayersTripl.TOB.skipClusters   = cms.InputTag('hiTobTecStepClusters')
tobTecStepSeedLayersTripl.MTOB.skipClusters   = cms.InputTag('hiTobTecStepClusters')
tobTecStepSeedLayersTripl.MTEC.skipClusters   = cms.InputTag('hiTobTecStepClusters')

# Triplet TrackingRegion
from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
hiTobTecStepTrackingRegionsTripl = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
     precise = True,
     useMultipleScattering = False,
     beamSpot = "offlineBeamSpot",
     useFoundVertices = True,
     useFakeVertices       = False,
     VertexCollection = "hiSelectedPixelVertex",
     useFixedError = True,
     fixedError = 5.0,#20.0
     ptMin = 0.9,#0.55
     originRadius = 3.5,
     originRScaling4BigEvts = cms.bool(True),
     halfLengthScaling4BigEvts = cms.bool(False),
     ptMinScaling4BigEvts = cms.bool(True),
     minOriginR = 0,
     minHalfLength = 0,
     maxPtMin = 1.2,#0.85
     scalingStartNPix = 20000,
     scalingEndNPix = 35000     
))

# Triplet seeding
tobTecStepHitDoubletsTripl.clusterCheck = ""
tobTecStepHitDoubletsTripl.trackingRegions = "hiTobTecStepTrackingRegionsTripl"

tobTecStepSeedLayersPair.TOB.skipClusters   = cms.InputTag('hiTobTecStepClusters')
tobTecStepSeedLayersPair.TEC.skipClusters   = cms.InputTag('hiTobTecStepClusters')

# Pair TrackingRegion
hiTobTecStepTrackingRegionsPair = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
     precise = True,
     useMultipleScattering = False,
     beamSpot = "offlineBeamSpot",
     useFoundVertices = True,
     useFakeVertices       = False,
     VertexCollection = "hiSelectedPixelVertex",
     useFixedError = True,
     fixedError = 7.5,#30.0
     ptMin = 0.9,#0.6
     originRadius = 6.0,
     originRScaling4BigEvts = cms.bool(True),
     halfLengthScaling4BigEvts = cms.bool(False),
     ptMinScaling4BigEvts = cms.bool(True),
     minOriginR = 0,
     minHalfLength = 0,
     maxPtMin = 1.5,#0.9
     scalingStartNPix = 20000,
     scalingEndNPix = 35000     
))

# Pair seeds
tobTecStepHitDoubletsPair.clusterCheck = ""
tobTecStepHitDoubletsPair.trackingRegions = "hiTobTecStepTrackingRegionsPair"


# QUALITY CUTS DURING TRACK BUILDING (for inwardss and outwards track building steps)
from RecoTracker.IterativeTracking.TobTecStep_cff import _tobTecStepTrajectoryFilterBase
_tobTecStepTrajectoryFilterBase.minimumNumberOfHits = 5
_tobTecStepTrajectoryFilterBase.minPt = 0.85

# TRACK BUILDING

# MAKING OF TRACK CANDIDATES
tobTecStepTrackCandidates.clustersToSkip = cms.InputTag('hiTobTecStepClusters')

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiTobTecStepTracks = tobTecStepTracks.clone()


# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiTobTecStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src='hiTobTecStepTracks',
    useAnyMVA = cms.bool(False),
    GBRForestLabel = cms.string('HIMVASelectorIter13'),
    GBRForestVars = cms.vstring(['chi2perdofperlayer', 'nhits', 'nlayers', 'eta']),
    trackSelectors= cms.VPSet(
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
    name = 'hiTobTecStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    ), #end of pset
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
    name = 'hiTobTecStepTight',
    preFilterName = 'hiTobTecStepLoose',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.2)
    ),
    RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
    name = 'hiTobTecStep',
    preFilterName = 'hiTobTecStepTight',
    applyAdaptedPVCuts = cms.bool(False),
    useMVA = cms.bool(False),
    minMVA = cms.double(-0.09)
    ),
    ) #end of vpset
    ) #end of clone

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
hiTobTecStepQual = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers=cms.VInputTag(cms.InputTag('hiTobTecStepTracks')),
    hasSelector=cms.vint32(1),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("hiTobTecStepSelector","hiTobTecStep")),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False),
    )


hiTobTecStep = cms.Sequence(hiTobTecStepClusters*
                          tobTecStepSeedLayersTripl*
                          hiTobTecStepTrackingRegionsTripl*
                          tobTecStepHitDoubletsTripl*
                          tobTecStepHitTripletsTripl*
                          tobTecStepSeedsTripl*
                          tobTecStepSeedLayersPair*
                          hiTobTecStepTrackingRegionsPair*
                          tobTecStepHitDoubletsPair*
                          tobTecStepSeedsPair*
                          tobTecStepSeeds*
                          tobTecStepTrackCandidates*
                          hiTobTecStepTracks*
                          hiTobTecStepSelector*
                          hiTobTecStepQual
                          )