import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 6th step: very large impact parameter trackng using TOB+TEC ring 5 seeding --pair

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaPhi      = 0.2
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaEta      = 0.1

###################################
from RecoTracker.IterativeTracking.TobTecStep_cff import *

# remove previously used clusters
hiRegitMuTobTecStepClusters = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepClusters.clone(
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuPixelLessStepClusters"),
    trajectories     = cms.InputTag("hiRegitMuPixelLessStepTracks"),
    overrideTrkQuals = cms.InputTag('hiRegitMuPixelLessStepSelector','hiRegitMuPixelLessStep'),
    trackClassifier       = cms.InputTag(''),
    TrackQuality          = cms.string('tight')
)

# SEEDING LAYERS
hiRegitMuTobTecStepSeedLayersPair =  RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedLayersPair.clone()
hiRegitMuTobTecStepSeedLayersPair.TOB.skipClusters = cms.InputTag('hiRegitMuTobTecStepClusters')
hiRegitMuTobTecStepSeedLayersPair.TEC.skipClusters = cms.InputTag('hiRegitMuTobTecStepClusters')

hiRegitMuTobTecStepSeedLayersTripl =  RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedLayersTripl.clone()
hiRegitMuTobTecStepSeedLayersTripl.TOB.skipClusters = cms.InputTag('hiRegitMuTobTecStepClusters')
hiRegitMuTobTecStepSeedLayersTripl.MTOB.skipClusters = cms.InputTag('hiRegitMuTobTecStepClusters')
hiRegitMuTobTecStepSeedLayersTripl.MTEC.skipClusters = cms.InputTag('hiRegitMuTobTecStepClusters')

# seeding
hiRegitMuTobTecStepSeeds     = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeeds.clone(
      seedCollections = cms.VInputTag(cms.InputTag("hiRegitMuTobTecStepSeedsTripl"), cms.InputTag("hiRegitMuTobTecStepSeedsPair"))
      )

# For now, keep the same parameters for triplets and pairs
hiRegitMuTobTecStepSeedsTripl     = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsTripl.clone()
hiRegitMuTobTecStepSeedsTripl.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuTobTecStepSeedsTripl.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuTobTecStepSeedsTripl.RegionFactoryPSet.MuonTrackingRegionBuilder.Pt_min          = 2.0
hiRegitMuTobTecStepSeedsTripl.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.2 # default = 0.2
hiRegitMuTobTecStepSeedsTripl.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ          = 0.2 # this give you the length 
hiRegitMuTobTecStepSeedsTripl.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuTobTecStepSeedsTripl.OrderedHitsFactoryPSet.SeedingLayers                        = 'hiRegitMuTobTecStepSeedLayersTripl'

hiRegitMuTobTecStepSeedsPair     = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepSeedsPair.clone()
hiRegitMuTobTecStepSeedsPair.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuTobTecStepSeedsPair.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuTobTecStepSeedsPair.RegionFactoryPSet.MuonTrackingRegionBuilder.Pt_min          = 2.0
hiRegitMuTobTecStepSeedsPair.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.2 # default = 0.2
hiRegitMuTobTecStepSeedsPair.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ          = 0.2 # this give you the length 
hiRegitMuTobTecStepSeedsPair.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuTobTecStepSeedsPair.OrderedHitsFactoryPSet.SeedingLayers                        = 'hiRegitMuTobTecStepSeedLayersPair'

# building: feed the new-named seeds
hiRegitMuTobTecStepInOutTrajectoryFilter = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepInOutTrajectoryFilter.clone()
hiRegitMuTobTecStepInOutTrajectoryFilter.minPt = 1.7
hiRegitMuTobTecStepInOutTrajectoryFilter.minimumNumberOfHits = 6
hiRegitMuTobTecStepInOutTrajectoryFilter.minHitsMinPt        = 4


hiRegitMuTobTecStepTrajectoryFilter = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepTrajectoryFilter.clone()
hiRegitMuTobTecStepTrajectoryFilter.minPt               = 1.7
hiRegitMuTobTecStepTrajectoryFilter.minimumNumberOfHits = 6
hiRegitMuTobTecStepTrajectoryFilter.minHitsMinPt        = 4   

hiRegitMuTobTecStepTrajectoryBuilder = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepTrajectoryBuilder.clone(
    trajectoryFilter      = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuTobTecStepTrajectoryFilter')
       ),
    inOutTrajectoryFilter = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuTobTecStepInOutTrajectoryFilter')
       ),
)

hiRegitMuTobTecStepTrackCandidates        =  RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuTobTecStepSeeds'),
    TrajectoryBuilderPSet = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuTobTecStepTrajectoryBuilder')
       ),
    clustersToSkip            = cms.InputTag('hiRegitMuTobTecStepClusters'),
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuTobTecStepTracks                 = RecoTracker.IterativeTracking.TobTecStep_cff.tobTecStepTracks.clone(
    AlgorithmName = cms.string('undefAlgorithm'),
    src                 = 'hiRegitMuTobTecStepTrackCandidates'
)

# import RecoHI.HiTracking.hiMultiTrackSelector_cfi
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
hiRegitMuTobTecStepSelector               = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone( 
    src                 ='hiRegitMuTobTecStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuTobTecStepLoose',
           qualityBit = cms.string('loose'),
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuTobTecStepTight',
            preFilterName = 'hiRegitMuTobTecStepLoose',
            qualityBit = cms.string('loose'),
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuTobTecStep',
            preFilterName = 'hiRegitMuTobTecStepTight',
            qualityBit = cms.string('tight'),
            ),
        ) #end of vpset
  
)

hiRegitMuonTobTecStep = cms.Sequence(hiRegitMuTobTecStepClusters*
                                     hiRegitMuTobTecStepSeedLayersTripl*
                                     hiRegitMuTobTecStepSeedsTripl*
                                     hiRegitMuTobTecStepSeedLayersPair*
                                     hiRegitMuTobTecStepSeedsPair*
                                     hiRegitMuTobTecStepSeeds*
                                     hiRegitMuTobTecStepTrackCandidates*
                                     hiRegitMuTobTecStepTracks*
                                     hiRegitMuTobTecStepSelector)



