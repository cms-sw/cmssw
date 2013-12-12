import FWCore.ParameterSet.Config as cms

from RecoTracker.ConversionSeedGenerators.PhotonConversionTrajectorySeedProducerFromQuadruplets_cfi import *

from RecoTracker.TrackProducer.TrackRefitters_cff                          import *
TrackRefitterStd = TrackRefitter.clone(src=cms.InputTag("generalTracks"))


convFilter = cms.EDProducer("QualityFilter",
                          TrackQuality = cms.string('loose'),
                          recTracks = cms.InputTag("TrackRefitterStd")
                          )

convClusters = cms.EDProducer("TrackClusterRemover",
                              trajectories = cms.InputTag("convFilter"),
                              pixelClusters = cms.InputTag("siPixelClusters"),
                              stripClusters = cms.InputTag("siStripClusters"),
                              Common = cms.PSet(maxChi2 = cms.double(30.0))
                              )

import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
convPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = 'convClusters'
    )

import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
convStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'convClusters'
    )

convLayerPairs = cms.ESProducer("SeedingLayersESProducer",
                                ComponentName = cms.string('convLayerPairs'),
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
                                    hitErrorRZ = cms.double(0.006),
                                    hitErrorRPhi = cms.double(0.0027),
                                    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
                                    HitProducer = cms.string('convPixelRecHits'),
                                    useErrorsFromParam = cms.bool(True)
                                    ),
                                FPix = cms.PSet(
                                    hitErrorRZ = cms.double(0.0036),
                                    hitErrorRPhi = cms.double(0.0051),
                                    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
                                    HitProducer = cms.string('convPixelRecHits'),
                                    useErrorsFromParam = cms.bool(True)
                                    ),
                                TIB1 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    matchedRecHits = cms.InputTag("convStripRecHits","matchedRecHit"),
                                    #useSimpleRphiHitsCleaner = cms.bool(False),
                                    #stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
                                    #rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched")
                                    ),
                                TIB2 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    matchedRecHits = cms.InputTag("convStripRecHits","matchedRecHit"),
                                    #useSimpleRphiHitsCleaner = cms.bool(False),
                                    #stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
                                    #rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched")
                                    ),
                                TIB3 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("convStripRecHits","rphiRecHit")
                                    ),
                                TIB4 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("convStripRecHits","rphiRecHit")
                                    ),
                                TID1 = cms.PSet(
                                    useSimpleRphiHitsCleaner = cms.bool(False),
                                    stereoRecHits = cms.InputTag("convStripRecHits","stereoRecHitUnmatched"),
                                    matchedRecHits = cms.InputTag("convStripRecHits","matchedRecHit"),
                                    useRingSlector = cms.bool(True),
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("convStripRecHits","rphiRecHitUnmatched"),
                                    maxRing = cms.int32(3),
                                    minRing = cms.int32(1)
                                    ),
                                TID2 = cms.PSet(
                                    useSimpleRphiHitsCleaner = cms.bool(False),
                                    stereoRecHits = cms.InputTag("convStripRecHits","stereoRecHitUnmatched"),
                                    matchedRecHits = cms.InputTag("convStripRecHits","matchedRecHit"),
                                    useRingSlector = cms.bool(True),
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("convStripRecHits","rphiRecHitUnmatched"),
                                    maxRing = cms.int32(3),
                                    minRing = cms.int32(1)
                                    ),
                                TID3 = cms.PSet(
                                    useSimpleRphiHitsCleaner = cms.bool(False),
                                    stereoRecHits = cms.InputTag("convStripRecHits","stereoRecHitUnmatched"),
                                    matchedRecHits = cms.InputTag("convStripRecHits","matchedRecHit"),
                                    useRingSlector = cms.bool(True),
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("convStripRecHits","rphiRecHitUnmatched"),
                                    maxRing = cms.int32(2),
                                    minRing = cms.int32(1)
                                    ),
                                TEC = cms.PSet(
                                    useSimpleRphiHitsCleaner = cms.bool(False),
                                    minRing = cms.int32(1),
                                    matchedRecHits = cms.InputTag("convStripRecHits","matchedRecHit"),
                                    useRingSlector = cms.bool(True),
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("convStripRecHits","rphiRecHitUnmatched"),
                                    maxRing = cms.int32(2),
                                    stereoRecHits = cms.InputTag("convStripRecHits","stereoRecHitUnmatched")
                                    ),
                                TOB1 = cms.PSet(
                                    matchedRecHits = cms.InputTag("convStripRecHits","matchedRecHit"),
                                    TTRHBuilder = cms.string('WithTrackAngle')
                                    ),
                                TOB2 = cms.PSet(
                                    matchedRecHits = cms.InputTag("convStripRecHits","matchedRecHit"),
                                    TTRHBuilder = cms.string('WithTrackAngle')
                                    ),
                                TOB3 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("convStripRecHits","rphiRecHit")
                                    ),
                                TOB4 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("convStripRecHits","rphiRecHit")
                                    ),
                                TOB5 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("convStripRecHits","rphiRecHit")
                                    ),
                                TOB6 = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    rphiRecHits = cms.InputTag("convStripRecHits","rphiRecHit")
                                    )
                                )


photonConvTrajSeedFromQuadruplets.TrackRefitter = cms.InputTag('TrackRefitterStd')

# TRACKER DATA CONTROL
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
convMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'convMeasurementTracker',
    pixelClusterProducer = 'convClusters',
    stripClusterProducer = 'convClusters'
    )

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
    MeasurementTrackerName = 'convMeasurementTracker',
    trajectoryFilterName = 'convCkfTrajectoryFilter',
    minNrOfHitsForRebuild = 3
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
convTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('photonConvTrajSeedFromQuadruplets:convSeedCandidates'),
    TrajectoryBuilder = 'convCkfTrajectoryBuilder'
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
convStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'convTrackCandidates',
    clusterRemovalInfo = 'convClusters',
    AlgorithmName = cms.string('iter9')
    )
    
# TRACK SELECTION AND QUALITY FLAG SETTING.
import RecoTracker.FinalTrackSelectors.selectLoose_cfi
import RecoTracker.FinalTrackSelectors.selectTight_cfi
import RecoTracker.FinalTrackSelectors.selectHighPurity_cfi
import RecoTracker.FinalTrackSelectors.simpleTrackListMerger_cfi
#since they are equal I comment out the first
# convStepLoose = RecoTracker.FinalTrackSelectors.selectLoose_cfi.selectLoose.clone(
#     src = 'convStepTracks',
#     keepAllTracks = True,
#     copyExtras = False,
#     copyTrajectories = True,
#     chi2n_par = 2.,
#     res_par = ( 0.003, 0.001 ),
#     minNumberLayers = 3,
#     maxNumberLostLayers = 1,
#     minNumber3DLayers = 1,
#     d0_par1 = ( 5., 8.0 ),
#     dz_par1 = ( 5., 8.0 ),
#     d0_par2 = ( 5., 8.0 ),
#     dz_par2 = ( 5., 8.0 )
#     )
# convStepTight = RecoTracker.FinalTrackSelectors.selectTight_cfi.selectTight.clone(
#     src = 'convStepLoose',
#     keepAllTracks = True,
#     copyExtras = False,
#     copyTrajectories = True,
#     chi2n_par = 2.,
#     res_par = ( 0.003, 0.001 ),
#     minNumberLayers = 3,
#     maxNumberLostLayers = 1,
#     minNumber3DLayers = 1,
#     d0_par1 = ( 5., 8.0 ),
#     dz_par1 = ( 5., 8.0 ),
#     d0_par2 = ( 5., 8.0 ),
#     dz_par2 = ( 5., 8.0 )
#     )
convStep = RecoTracker.FinalTrackSelectors.selectHighPurity_cfi.selectHighPurity.clone(
#    src = 'convStepTight',
    src = 'convStepTracks',
    keepAllTracks = True,
    copyExtras = False,
    copyTrajectories = True,
    applyAdaptedPVCuts = False,
    chi2n_par = 2.,
    res_par = ( 0.003, 0.001 ),
    minNumberLayers = 3,
    maxNumberLostLayers = 1,
    minNumber3DLayers = 1,
    d0_par1 = ( 5., 8.0 ),
    dz_par1 = ( 5., 8.0 ),
    d0_par2 = ( 5., 8.0 ),
    dz_par2 = ( 5., 8.0 )
    )

#import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
#QuadSeedStep = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
#    TrackProducers = cms.VInputTag(cms.InputTag('convStepTracks')),
#    hasSelector=cms.vint32(1),
#    selectedTrackQuals = cms.VInputTag(cms.InputTag("convStep")),
#    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0), pQual=cms.bool(True) )),
#    writeOnlyTrkQuals=cms.bool(True)
#)

convSequence = cms.Sequence( TrackRefitterStd * convFilter * convClusters * convPixelRecHits * convStripRecHits 
                             * photonConvTrajSeedFromQuadruplets 
                             *convTrackCandidates*convStepTracks
                             #*convStepLoose*convStepTight
                             *convStep
                             #*QuadSeedStep
                             )




