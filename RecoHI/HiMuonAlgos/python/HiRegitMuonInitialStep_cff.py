import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 0st step:pixel-triplet seeding, high-pT;
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.2
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2


###################################  
from RecoTracker.IterativeTracking.InitialStep_cff import *

# seeding
hiRegitMuInitialStepSeeds     = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.clone()
hiRegitMuInitialStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuInitialStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 1.5
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 0.03 # default = 0.2
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 0. # this give you the length 
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())


# building: feed the new-named seeds
hiRegitMuInitialStepTrajectoryFilter = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrajectoryFilter.clone(
    ComponentName = 'hiRegitMuInitialStepTrajectoryFilter'
    )
hiRegitMuInitialStepTrajectoryFilter.filterPset.minPt              = 1.4 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),


hiRegitMuInitialStepTrajectoryBuilder = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuInitialStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuInitialStepTrajectoryFilter'
)

# track candidates
hiRegitMuInitialStepTrackCandidates        =  RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuInitialStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuInitialStepTrajectoryBuilder'
    )

# fitting: feed new-names
hiRegitMuInitialStepTracks                 = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTracks.clone(
    src                 = 'hiRegitMuInitialStepTrackCandidates'
)


hiRegitMuInitialStepSelector               = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSelector.clone( 
    src                 ='hiRegitMuInitialStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'hiRegitMuInitialStepLoose',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuInitialStepTight',
            preFilterName = 'hiRegitMuInitialStepLoose',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuInitialStep',
            preFilterName = 'hiRegitMuInitialStepTight',
            ),
        ) #end of vpset
    )

hiRegitMuonInitialStep = cms.Sequence(hiRegitMuInitialStepSeeds*
                                      hiRegitMuInitialStepTrackCandidates*
                                      hiRegitMuInitialStepTracks*
                                      hiRegitMuInitialStepSelector)

