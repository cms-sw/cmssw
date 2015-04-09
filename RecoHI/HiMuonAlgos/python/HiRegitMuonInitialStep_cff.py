import FWCore.ParameterSet.Config as cms

# pp iterative tracking modified for hiOffline reco (the vertex is the one reconstructed in HI)
################################### 0st step:pixel-triplet seeding, high-pT;
from RecoHI.HiTracking.HITrackingRegionProducer_cfi import *
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.vertexCollection = cms.InputTag("hiSelectedVertex")
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonSrc= cms.InputTag("standAloneMuons","UpdatedAtVtx")

HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Phi_fixed     = True 
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.Eta_fixed     = True
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaPhi      = 0.3
HiTrackingRegionFactoryFromSTAMuonsBlock.MuonTrackingRegionBuilder.DeltaEta      = 0.2

###################################  
from RecoTracker.IterativeTracking.InitialStep_cff import *

# SEEDING LAYERS
hiRegitMuInitialStepSeedLayers =  RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeedLayers.clone()

# seeding
hiRegitMuInitialStepSeeds     = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSeeds.clone()
hiRegitMuInitialStepSeeds.RegionFactoryPSet                                           = HiTrackingRegionFactoryFromSTAMuonsBlock.clone()
hiRegitMuInitialStepSeeds.OrderedHitsFactoryPSet.SeedingLayers                        = cms.InputTag("hiRegitMuInitialStepSeedLayers")
hiRegitMuInitialStepSeeds.ClusterCheckPSet.doClusterCheck                             = False # do not check for max number of clusters pixel or strips
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.EscapePt        = 3.0
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaR          = 1 # default = 0.2
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.DeltaZ          = 1 # this give you the length 
hiRegitMuInitialStepSeeds.RegionFactoryPSet.MuonTrackingRegionBuilder.Rescale_Dz      = 4. # max(DeltaZ_Region,Rescale_Dz*vtx->zError())


# building: feed the new-named seeds
hiRegitMuInitialStepTrajectoryFilterBase = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrajectoryFilterBase.clone()
hiRegitMuInitialStepTrajectoryFilterBase.minPt = 2.5 # after each new hit, apply pT cut for traj w/ at least minHitsMinPt = cms.int32(3),

hiRegitMuInitialStepTrajectoryFilter = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrajectoryFilter.clone()
hiRegitMuInitialStepTrajectoryFilter.filters = cms.VPSet(
      cms.PSet( refToPSet_ = cms.string('hiRegitMuInitialStepTrajectoryFilterBase')),
      cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterShape')))


hiRegitMuInitialStepTrajectoryBuilder = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuInitialStepTrajectoryFilter')
       ),
)

# track candidates
hiRegitMuInitialStepTrackCandidates        =  RecoTracker.IterativeTracking.InitialStep_cff.initialStepTrackCandidates.clone(
    src               = cms.InputTag('hiRegitMuInitialStepSeeds'),
    TrajectoryBuilderPSet = cms.PSet(
       refToPSet_ = cms.string('hiRegitMuInitialStepTrajectoryBuilder')
       ),
    maxNSeeds         = cms.uint32(1000000)
    )

# fitting: feed new-names
hiRegitMuInitialStepTracks                 = RecoTracker.IterativeTracking.InitialStep_cff.initialStepTracks.clone(
    AlgorithmName = cms.string('iter3'),
    src                 = 'hiRegitMuInitialStepTrackCandidates'
)


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
hiRegitMuInitialStepSelector               = RecoTracker.IterativeTracking.InitialStep_cff.initialStepSelector.clone( 
    src                 ='hiRegitMuInitialStepTracks',
    vertices            = cms.InputTag("hiSelectedVertex"),
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
           name = 'hiRegitMuInitialStepLoose',
           qualityBit = cms.string('loose'),
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'hiRegitMuInitialStepTight',
            preFilterName = 'hiRegitMuInitialStepLoose',
            qualityBit = cms.string('loose'),
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'hiRegitMuInitialStep',
            preFilterName = 'hiRegitMuInitialStepTight',
            qualityBit = cms.string('tight'),
            ),
        ) #end of vpset
    )

hiRegitMuonInitialStep = cms.Sequence(hiRegitMuInitialStepSeedLayers*
                                      hiRegitMuInitialStepSeeds*
                                      hiRegitMuInitialStepTrackCandidates*
                                      hiRegitMuInitialStepTracks*
                                      hiRegitMuInitialStepSelector)

