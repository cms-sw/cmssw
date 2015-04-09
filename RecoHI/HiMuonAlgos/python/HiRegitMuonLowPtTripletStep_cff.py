import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 1st step:pixel-triplet seeding, lower-pT;

from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaPhi      = 0.3
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaEta      = 0.2

###################################
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *

# remove previously used clusters
hiRegitMuLowPtTripletStepClusters = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepClusters.clone(
    trajectories          = cms.InputTag("hiRegitMuDetachedTripletStepTracks"),
    overrideTrkQuals      = cms.InputTag('hiRegitMuDetachedTripletStepSelector','hiRegitMuDetachedTripletStep'),
    oldClusterRemovalInfo = cms.InputTag("hiRegitMuDetachedTripletStepClusters"),
    TrackQuality          = cms.string('tight')
)

# SEEDING LAYERS
hiRegitMuLowPtTripletStepSeedLayers =  RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeedLayers.clone()
hiRegitMuLowPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('hiRegitMuLowPtTripletStepClusters')
hiRegitMuLowPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('hiRegitMuLowPtTripletStepClusters')

# seeds
hiRegitMuLowPtTripletStepSeeds     = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSeeds.clone()
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuLowPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 0.9
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 1. # default = 0.2
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ          = 1. # this give you the length 
hiRegitMuLowPtTripletStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuLowPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers                                  = 'hiRegitMuLowPtTripletStepSeedLayers'



# building: feed the new-named seeds
hiRegitMuLowPtTripletStepStandardTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
      minPt = 0.8
      )

hiRegitMuLowPtTripletStepTrajectoryFilter = cms.PSet(
      ComponentType = cms.string('CompositeTrajectoryFilter'),
      filters = cms.VPSet(cms.PSet(
         refToPSet_ = cms.string('hiRegitMuLowPtTripletStepStandardTrajectoryFilter')
         ), 
         cms.PSet(
            refToPSet_ = cms.string('ClusterShapeTrajectoryFilter')
            ))
         )


hiRegitMuLowPtTripletStepTrajectoryBuilder = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuLowPtTripletStepTrajectoryFilter',)
       ),
)

# track candidates
hiRegitMuLowPtTripletStepTrackCandidates        =  RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuLowPtTripletStepSeeds'),
    TrajectoryBuilderPSet = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuLowPtTripletStepTrajectoryBuilder')
       ),
    clustersToSkip = cms.InputTag('hiRegitMuLowPtTripletStepClusters'),
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuLowPtTripletStepTracks                 = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepTracks.clone(
    AlgorithmName = cms.string('iter4'),
    src                 = 'hiRegitMuLowPtTripletStepTrackCandidates'
)


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
hiRegitMuLowPtTripletStepSelector               = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepSelector.clone( 
    src                 ='hiRegitMuLowPtTripletStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuLowPtTripletStepLoose',
           qualityBit = cms.string('loose'),
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuLowPtTripletStepTight',
            preFilterName = 'hiRegitMuLowPtTripletStepLoose',
            qualityBit = cms.string('loose'),
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuLowPtTripletStep',
            preFilterName = 'hiRegitMuLowPtTripletStepTight',
            qualityBit = cms.string('tight'),
            ),
        ) #end of vpset
)

hiRegitMuonLowPtTripletStep = cms.Sequence(hiRegitMuLowPtTripletStepClusters*
                                           hiRegitMuLowPtTripletStepSeedLayers*
                                           hiRegitMuLowPtTripletStepSeeds*
                                           hiRegitMuLowPtTripletStepTrackCandidates*
                                           hiRegitMuLowPtTripletStepTracks*
                                           hiRegitMuLowPtTripletStepSelector)



