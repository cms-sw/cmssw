import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 0st step:pixel-triplet seeding, high-pT;
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *

###################################  
from RecoTracker.IterativeTracking.InitialStep_cff import *

#################################
# Filter on quality tracks
hiGeneralTrackFilter = cms.EDProducer("QualityFilter",
                                      TrackQuality = cms.string('highPurity'),
                                      recTracks = cms.InputTag("hiGeneralTracks")
                                      )

# NEW CLUSTERS (remove previously used clusters)
hiRegitInitialStepClusters = cms.EDProducer("HITrackClusterRemover",
                                            clusterLessSolution= cms.bool(True),
                                            oldClusterRemovalInfo = cms.InputTag("hiPixelPairClusters"),
                                            trajectories = cms.InputTag("hiGeneralTrackFilter"),
                                            TrackQuality = cms.string('highPurity'),
                                            pixelClusters = cms.InputTag("siPixelClusters"),
                                            stripClusters = cms.InputTag("siStripClusters"),
                                            Common = cms.PSet(
						    maxChi2 = cms.double(9.0),
						    ),
					    Strip = cms.PSet(
						    maxChi2 = cms.double(9.0),
						    )
                                            )



# seeding
hiRegitInitialStepSeeds     = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.clone(
    RegionFactoryPSet = HiTrackingRegionFactoryFromJetsBlock.clone(
	RegionPSet = dict(ptMin = 1.2)
    ),
    ClusterCheckPSet = dict(doClusterCheck = False), # do not check for max number of clusters pixel or strips
    skipClusters = cms.InputTag('hiRegitInitialStepClusters')
)
# building: feed the new-named seeds
hiRegitInitialStepTrajectoryFilter = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrajectoryFilterBase.clone()


hiRegitInitialStepTrajectoryBuilder = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('hiRegitInitialStepTrajectoryFilter')),
    clustersToSkip = cms.InputTag('hiRegitInitialStepClusters')
)

# track candidates
hiRegitInitialStepTrackCandidates        =  RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrackCandidates.clone(
    src               = 'hiRegitInitialStepSeeds',
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('hiRegitInitialStepTrajectoryBuilder')),
    maxNSeeds = 100000
)

# fitting: feed new-names
hiRegitInitialStepTracks                 = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTracks.clone(
    src                 = 'hiRegitInitialStepTrackCandidates',
    AlgorithmName = 'initialStep'
)


# Track selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiRegitInitialStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src = 'hiRegitInitialStepTracks',
    trackSelectors = cms.VPSet(
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
           name = 'hiRegitInitialStepLoose',
           d0_par2 = [9999.0, 0.0],
           dz_par2 = [9999.0, 0.0],
           applyAdaptedPVCuts = False
       ), #end of pset
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
           name = 'hiRegitInitialStepTight',
           preFilterName = 'hiRegitInitialStepLoose',
           d0_par2 = [9999.0, 0.0],
           dz_par2 = [9999.0, 0.0],
           applyAdaptedPVCuts = False
       ),
       RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
           name = 'hiRegitInitialStep',
           preFilterName = 'hiRegitInitialStepTight',
           d0_par2 = [9999.0, 0.0],
           dz_par2 = [9999.0, 0.0],
           applyAdaptedPVCuts = False
       ),
    ) #end of vpset
) #end of clone  


hiRegitInitialStepTask = cms.Task(hiGeneralTrackFilter,
                                  hiRegitInitialStepClusters,
                                  hiRegitInitialStepSeeds,
                                  hiRegitInitialStepTrackCandidates,
                                  hiRegitInitialStepTracks,
                                  hiRegitInitialStepSelector)
hiRegitInitialStep = cms.Sequence(hiRegitInitialStepTask)
