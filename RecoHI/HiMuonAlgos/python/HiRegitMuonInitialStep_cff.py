import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 0st step:pixel-triplet seeding, high-pT;
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseVertex      = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.UseFixedRegion = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed      = 0.3
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed      = 0.2


###################################  
hiRegitMuFirstStepFilter = cms.EDProducer("QualityFilter",
                                 TrackQuality = cms.string('highPurity'),
                                 recTracks = cms.InputTag("hiSelectedTracks")
                                 )

# NEW CLUSTERS (remove previously used clusters)
hiRegitMuInitialStepClusters = cms.EDProducer("TrackClusterRemover",
                                clusterLessSolution= cms.bool(True),
                                trajectories = cms.InputTag("hiRegitMuFirstStepFilter"),
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

#-------------------------
from RecoHI.HiTracking.hiRegitInitialStep_cff import *

# seeding
hiRegitMuInitialStepSeeds     =  RecoHI.HiTracking.hiRegitInitialStep_cff.hiRegitInitialStepSeeds.clone()
hiRegitMuInitialStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuInitialStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 3.0
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 1 # default = 0.2
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ_Region   = 1 # this give you the length 
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())
hiRegitMuInitialStepSeeds.skipClusters = cms.InputTag('hiRegitMuInitialStepClusters')



# building: feed the new-named seeds
hiRegitMuInitialStepTrajectoryFilter =  RecoHI.HiTracking.hiRegitInitialStep_cff.hiRegitInitialStepTrajectoryFilter.clone(
    ComponentName = 'hiRegitMuInitialStepTrajectoryFilter'
    )
hiRegitMuInitialStepTrajectoryFilter.filterPset.minPt = 2.5 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),


hiRegitMuInitialStepTrajectoryBuilder =  RecoHI.HiTracking.hiRegitInitialStep_cff.hiRegitInitialStepTrajectoryBuilder.clone(
    ComponentName        = 'hiRegitMuInitialStepTrajectoryBuilder',
    trajectoryFilterName = 'hiRegitMuInitialStepTrajectoryFilter',
    clustersToSkip       = cms.InputTag('hiRegitMuInitialStepClusters')
)

# track candidates
hiRegitMuInitialStepTrackCandidates        =   RecoHI.HiTracking.hiRegitInitialStep_cff.hiRegitInitialStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuInitialStepSeeds'),
    TrajectoryBuilder = 'hiRegitMuInitialStepTrajectoryBuilder',
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuInitialStepTracks                 =  RecoHI.HiTracking.hiRegitInitialStep_cff.hiRegitInitialStepTracks.clone(
    src                 = 'hiRegitMuInitialStepTrackCandidates'
)


hiRegitMuInitialStepSelector               =  RecoHI.HiTracking.hiRegitInitialStep_cff.hiRegitInitialStepSelector.clone( 
    src                 ='hiRegitMuInitialStepTracks',
    trackSelectors= cms.VPSet(
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
            name = 'hiRegitMuInitialStepLoose',
            ), #end of pset
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiRegitMuInitialStepTight',
            preFilterName = 'hiRegitMuInitialStepLoose',
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiRegitMuInitialStep',
            preFilterName = 'hiRegitMuInitialStepTight',
#            minNumberLayers = 10
            ),
        ) #end of vpset
    )

hiRegitMuonInitialStep = cms.Sequence(hiRegitMuFirstStepFilter*
                                      hiRegitMuInitialStepClusters*
                                      hiRegitMuInitialStepSeeds*
                                      hiRegitMuInitialStepTrackCandidates*
                                      hiRegitMuInitialStepTracks*
                                      hiRegitMuInitialStepSelector)

