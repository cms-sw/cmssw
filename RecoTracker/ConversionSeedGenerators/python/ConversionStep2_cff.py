import FWCore.ParameterSet.Config as cms

from RecoTracker.ConversionSeedGenerators.PhotonConversionTrajectorySeedProducerFromQuadruplets_cfi import *

conv2Clusters = cms.EDProducer("TrackClusterRemover",
                              clusterLessSolution = cms.bool(True),
                              oldClusterRemovalInfo = cms.InputTag("convClusters"),
                              trajectories = cms.InputTag("convStepTracks"),
                              overrideTrkQuals = cms.InputTag('convStepSelector','convStep'),
                              TrackQuality = cms.string('highPurity'),
                              pixelClusters = cms.InputTag("siPixelClusters"),
                              stripClusters = cms.InputTag("siStripClusters"),
                              Common = cms.PSet(maxChi2 = cms.double(30.0))
                              )

conv2LayerPairs = cms.EDProducer("SeedingLayersEDProducer",
                                layerList = cms.vstring('BPix1+BPix2', 

                                                        'BPix2+BPix3', 
                                                        'BPix2+FPix1_pos', 
                                                        'BPix2+FPix1_neg', 
                                                        'BPix2+FPix2_pos', 
                                                        'BPix2+FPix2_neg', 

                                                        'FPix1_pos+FPix2_pos', 
                                                        'FPix1_neg+FPix2_neg',

                                                        'BPix3+TIB1', 
                                                        
                                                        'TIB1+TID1_pos', 
                                                        'TIB1+TID1_neg', 
                                                        'TIB1+TID2_pos', 
                                                        'TIB1+TID2_neg',
                                                        'TIB1+TIB2',
                                                      
                                                        'TIB2+TID1_pos', 
                                                        'TIB2+TID1_neg', 
                                                        'TIB2+TID2_pos', 
                                                        'TIB2+TID2_neg', 
                                                        'TIB2+TIB3',
                                                      
                                                        'TIB3+TIB4', 
                                                        'TIB3+TID1_pos', 
                                                        'TIB3+TID1_neg', 

                                                        'TIB4+TOB1',

                                                        'TOB1+TOB2', 
                                                        'TOB1+TEC1_pos', 
                                                        'TOB1+TEC1_neg', 

                                                        'TOB2+TOB3',  
                                                        'TOB2+TEC1_pos', 
                                                        'TOB2+TEC1_neg', 
                                                        
                                                        'TOB3+TOB4', 
                                                        'TOB3+TEC1_pos', 
                                                        'TOB3+TEC1_neg', 
                                                        
                                                        'TOB4+TOB5',

                                                        'TOB5+TOB6',

                                                        'TID1_pos+TID2_pos', 
                                                        'TID2_pos+TID3_pos', 
                                                        'TID3_pos+TEC1_pos', 

                                                        'TID1_neg+TID2_neg', 
                                                        'TID2_neg+TID3_neg', 
                                                        'TID3_neg+TEC1_neg', 

                                                        'TEC1_pos+TEC2_pos', 
                                                        'TEC2_pos+TEC3_pos', 
                                                        'TEC3_pos+TEC4_pos',
                                                        'TEC4_pos+TEC5_pos',
                                                        'TEC5_pos+TEC6_pos',
                                                        'TEC6_pos+TEC7_pos',
                                                        'TEC7_pos+TEC8_pos',

                                                        'TEC1_neg+TEC2_neg', 
                                                        'TEC2_neg+TEC3_neg', 
                                                        'TEC3_neg+TEC4_neg',
                                                        'TEC4_neg+TEC5_neg',
                                                        'TEC5_neg+TEC6_neg',
                                                        'TEC6_neg+TEC7_neg',
                                                        'TEC7_neg+TEC8_neg'
                                                        #other combinations could be added
                                                        ),
                                
                                BPix = cms.PSet(
                                    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
                                    HitProducer = cms.string('siPixelRecHits'),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                FPix = cms.PSet(
                                    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
                                    HitProducer = cms.string('siPixelRecHits'),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TIB1 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TIB2 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TIB3 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TIB4 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TID1 = cms.PSet(
                                    useSimpleRphiHitsCleaner = cms.bool(False),
                                    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
                                    useRingSlector = cms.bool(True),
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    maxRing = cms.int32(2),
                                    minRing = cms.int32(1),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TID2 = cms.PSet(
                                    useSimpleRphiHitsCleaner = cms.bool(False),
                                    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
                                    useRingSlector = cms.bool(True),
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    maxRing = cms.int32(2),
                                    minRing = cms.int32(1),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TID3 = cms.PSet(
                                    useSimpleRphiHitsCleaner = cms.bool(False),
                                    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
                                    useRingSlector = cms.bool(True),
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    maxRing = cms.int32(2),
                                    minRing = cms.int32(1),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TEC = cms.PSet(
                                    useSimpleRphiHitsCleaner = cms.bool(False),
                                    minRing = cms.int32(1),
                                    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
                                    useRingSlector = cms.bool(True),
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
                                    maxRing = cms.int32(7),
                                    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TOB1 = cms.PSet(
                                    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TOB2 = cms.PSet(
                                    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TOB3 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TOB4 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TOB5 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    ),
                                TOB6 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
                                    skipClusters = cms.InputTag('conv2Clusters'),
                                    )
                                )


photonConvTrajSeedFromQuadruplets.TrackRefitter = cms.InputTag('generalTracks')
photonConvTrajSeedFromQuadruplets.primaryVerticesTag = cms.InputTag('pixelVertices')


# TRACKER DATA CONTROL

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
conv2CkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
        maxLostHits = 1,
        minimumNumberOfHits = 3,
        minPt = 0.1
    )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
conv2CkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('conv2CkfTrajectoryFilter')),
    minNrOfHitsForRebuild = 3,
    clustersToSkip = cms.InputTag('conv2Clusters'),
    maxCand = 2
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
conv2TrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('photonConvTrajSeedFromQuadruplets:conv2SeedCandidates'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('conv2CkfTrajectoryBuilder'))
)

import TrackingTools.TrackFitters.RungeKuttaFitters_cff
conv2StepFitterSmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'conv2StepFitterSmoother',
    EstimateCut = 30,
    Smoother = cms.string('conv2StepRKSmoother')
    )
    
conv2StepRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('conv2StepRKSmoother'),
    errorRescaling = 10.0
    )

        
# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
conv2StepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'conv2TrackCandidates',
    AlgorithmName = cms.string('iter9'),
    Fitter = 'conv2StepFitterSmoother',
    )


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
conv2StepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='conv2StepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'conv2StepLoose',
            applyAdaptedPVCuts = False,
            chi2n_par = 3.0,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 1,
            d0_par1 = ( 5., 8.0 ),
            dz_par1 = ( 5., 8.0 ),
            d0_par2 = ( 5., 8.0 ),
            dz_par2 = ( 5., 8.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'conv2StepTight',
            preFilterName = 'conv2StepLoose',
            chi2n_par = 2.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 1,
            d0_par1 = ( 5., 8.0 ),
            dz_par1 = ( 5., 8.0 ),
            d0_par2 = ( 5., 8.0 ),
            dz_par2 = ( 5., 8.0 )
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'conv2Step',
            preFilterName = 'conv2StepTight',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 1,
            minNumber3DLayers = 1,
            d0_par1 = ( 5., 8.0 ),
            dz_par1 = ( 5., 8.0 ),
            d0_par2 = ( 5., 8.0 ),
            dz_par2 = ( 5., 8.0 )
            ),
        ) #end of vpset
    ) #end of clone

Conv2Step = cms.Sequence( conv2Clusters 
                         + conv2LayerPairs
                         + photonConvTrajSeedFromQuadruplets 
                         + conv2TrackCandidates
                         + conv2StepTracks
                         + conv2StepSelector)
