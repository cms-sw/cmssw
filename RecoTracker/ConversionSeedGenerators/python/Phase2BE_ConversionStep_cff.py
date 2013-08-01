import FWCore.ParameterSet.Config as cms

from RecoTracker.ConversionSeedGenerators.Phase2BE_PhotonConversionTrajectorySeedProducerFromSingleLeg_cfi import *
from RecoTracker.ConversionSeedGenerators.ConversionStep2_cff import *

convClusters = cms.EDProducer("TrackClusterRemover",
                              clusterLessSolution = cms.bool(True),
                              oldClusterRemovalInfo = cms.InputTag("pixelPairStepClusters"),
                              trajectories = cms.InputTag("pixelPairStepTracks"),
                              overrideTrkQuals = cms.InputTag('pixelPairStepSelector','pixelPairStep'),
                              TrackQuality = cms.string('highPurity'),
                              pixelClusters = cms.InputTag("siPixelClusters"),
                              stripClusters = cms.InputTag("siStripClusters"),
                              Common = cms.PSet(maxChi2 = cms.double(30.0))
                              )

convLayerPairs = cms.ESProducer("SeedingLayersESProducer",
                                ComponentName = cms.string('convLayerPairs'),
                                layerList = cms.vstring('BPix1+BPix2', 
                                                        'BPix2+BPix3', 
                                                        'BPix3+BPix4', 

                                                        'BPix1+FPix1_pos', 
                                                        'BPix1+FPix1_neg', 
                                                        'BPix2+FPix1_pos', 
                                                        'BPix2+FPix1_neg', 
                                                        'BPix3+FPix1_pos', 
                                                        'BPix3+FPix1_neg', 

                                                        'FPix1_pos+FPix2_pos', 
                                                        'FPix1_neg+FPix2_neg',
                                                        'FPix2_pos+FPix3_pos', 
                                                        'FPix2_neg+FPix3_neg'
                                                        ),

                                BPix = cms.PSet(
                                    hitErrorRZ = cms.double(0.006),
                                    hitErrorRPhi = cms.double(0.0027),
                                    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
                                    HitProducer = cms.string('siPixelRecHits'),
                                    useErrorsFromParam = cms.bool(True),
                                    skipClusters = cms.InputTag('convClusters'),
                                    ),
                                FPix = cms.PSet(
                                    hitErrorRZ = cms.double(0.0036),
                                    hitErrorRPhi = cms.double(0.0051),
                                    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
                                    HitProducer = cms.string('siPixelRecHits'),
                                    useErrorsFromParam = cms.bool(True),
                                    skipClusters = cms.InputTag('convClusters'),
                                    )
                                )


photonConvTrajSeedFromSingleLeg.TrackRefitter = cms.InputTag('generalTracks')
photonConvTrajSeedFromSingleLeg.primaryVerticesTag = cms.InputTag('pixelVertices')
#photonConvTrajSeedFromQuadruplets.TrackRefitter = cms.InputTag('generalTracks')
#photonConvTrajSeedFromQuadruplets.primaryVerticesTag = cms.InputTag('pixelVertices')


# TRACKER DATA CONTROL

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
convCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'convCkfTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
        maxLostHits = 1,
        minimumNumberOfHits = 3,
        minPt = 0.1
        )
    )

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
convCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'convCkfTrajectoryBuilder',
    trajectoryFilterName = 'convCkfTrajectoryFilter',
    minNrOfHitsForRebuild = 3,
    clustersToSkip = cms.InputTag('convClusters'),
    maxCand = 2
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
convTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('photonConvTrajSeedFromSingleLeg:convSeedCandidates'),
    TrajectoryBuilder = 'convCkfTrajectoryBuilder'
)

import TrackingTools.TrackFitters.RungeKuttaFitters_cff
convStepFitterSmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'convStepFitterSmoother',
    EstimateCut = 30,
    Smoother = cms.string('convStepRKSmoother')
    )
    
convStepRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName = cms.string('convStepRKSmoother'),
    errorRescaling = 10.0
    )

        
# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
convStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'convTrackCandidates',
    AlgorithmName = cms.string('iter8'),
    Fitter = 'convStepFitterSmoother',
    TTRHBuilder=cms.string('WithTrackAngle')
    )


import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
convStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='convStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'convStepLoose',
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
            name = 'convStepTight',
            preFilterName = 'convStepLoose',
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
            name = 'convStep',
            preFilterName = 'convStepTight',
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

ConvStep = cms.Sequence( convClusters 
                         + photonConvTrajSeedFromSingleLeg 
                         + convTrackCandidates
                         + convStepTracks
                         + convStepSelector
                         #+ Conv2Step #full quad-seeding sequence
                         )


### Quad-seeding sequence disabled (#+ Conv2Step)
# if enabled, the quad-seeded tracks have to be merged with the single-leg seeded tracks
# in RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff change:
###
#conversionStepTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
#    TrackProducers = cms.VInputTag(cms.InputTag('convStepTracks')),
#    hasSelector=cms.vint32(1),
#    selectedTrackQuals = cms.VInputTag(cms.InputTag("convStepSelector","convStep")
#                                       ),
#    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(1), pQual=cms.bool(True) )
#                             ),
#    copyExtras = True,
#    makeReKeyedSeeds = cms.untracked.bool(False)
#    )
###
# TO this:
###
#conversionStepTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
#    TrackProducers = cms.VInputTag(
#                                   cms.InputTag('convStepTracks'),
#                                   cms.InputTag('conv2StepTracks')
#                                   ),
#    hasSelector=cms.vint32(1,1),
#    selectedTrackQuals = cms.VInputTag(
#                                       cms.InputTag("convStepSelector","convStep"),
#                                       cms.InputTag("conv2StepSelector","conv2Step")
#                                       ),
#    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1), pQual=cms.bool(True) )
#                             ),
#    copyExtras = True,
#    makeReKeyedSeeds = cms.untracked.bool(False)
#    )
###
