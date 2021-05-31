import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi import *

from RecoTracker.ConversionSeedGenerators.PhotonConversionTrajectorySeedProducerFromSingleLeg_cfi import *
from RecoTracker.ConversionSeedGenerators.ConversionStep2_cff import *

from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import trackClusterRemover as _trackClusterRemover
_convClustersBase = _trackClusterRemover.clone(
    maxChi2               = 30.0,
    trajectories          = 'tobTecStepTracks',
    pixelClusters         = 'siPixelClusters',
    stripClusters         = 'siStripClusters',
    oldClusterRemovalInfo = 'tobTecStepClusters',
    TrackQuality          = 'highPurity'
)

convClusters = _convClustersBase.clone(
    trackClassifier       = 'tobTecStep:QualityMasks',
)


#Phase2 : configuring the phase2 track Cluster Remover
from RecoLocalTracker.SubCollectionProducers.phase2trackClusterRemover_cfi import phase2trackClusterRemover as _phase2trackClusterRemover
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toReplaceWith(convClusters, _phase2trackClusterRemover.clone(
    maxChi2                                  = 30.0,
    phase2pixelClusters                      = 'siPixelClusters',
    phase2OTClusters                         = 'siPhase2Clusters',
    TrackQuality                             = 'highPurity',
    minNumberOfLayersWithMeasBeforeFiltering = 0,
    trajectories                             = 'detachedQuadStepTracks',
    oldClusterRemovalInfo                    = 'detachedQuadStepClusters',
    overrideTrkQuals                         = 'detachedQuadStepSelector:detachedQuadStepTrk'
))

_convLayerPairsStripOnlyLayers = ['TIB1+TID1_pos', 
                                 'TIB1+TID1_neg', 
                                 'TIB1+TID2_pos', 
                                 'TIB1+TID2_neg',
                                 'TIB1+TIB2',
                                 'TIB1+MTIB3',
                                 
                                 'TIB2+TID1_pos', 
                                 'TIB2+TID1_neg', 
                                 'TIB2+TID2_pos', 
                                 'TIB2+TID2_neg', 
                                 'TIB2+MTIB3',
                                 'TIB2+MTIB4',
                                 
                                 'MTIB3+MTIB4',
                                 'MTIB3+TOB1',
                                 'MTIB3+TID1_pos',
                                 'MTIB3+TID1_neg',
                                 
                                 'MTIB4+TOB1',
                                 'MTIB4+TOB2',
                                 
                                 'TOB1+TOB2',
                                 'TOB1+MTOB3',
                                 'TOB1+TEC1_pos',
                                 'TOB1+TEC1_neg',

                                 'TOB2+MTOB3',
                                 'TOB2+MTOB4',
                                 'TOB2+TEC1_pos',
                                 'TOB2+TEC1_neg',
                                 
                                 #NB: re-introduce these combinations when large displaced
                                 #    tracks, reconstructed only in TOB will be available
                                 #    For instance think at the OutIn Ecal Seeded tracks
                                 #'MTOB3+MTOB4',
                                 #'MTOB3+MTOB5',
                                 #'MTOB3+TEC1_pos',
                                 #'MTOB3+TEC1_neg',
                                 #
                                 #'MTOB4+MTOB5',
                                 #'MTOB4+MTOB6',
                                 #
                                 #'MTOB5+MTOB6',
                                 
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
                                 'TEC7_neg+TEC8_neg']
                    
_convLayerPairsLayerList = _convLayerPairsStripOnlyLayers
_convLayerPairsLayerList =['BPix1+BPix2', 
                           
                           'BPix2+BPix3', 
                           'BPix2+FPix1_pos', 
                           'BPix2+FPix1_neg', 
                           'BPix2+FPix2_pos', 
                           'BPix2+FPix2_neg', 
                           
                           'FPix1_pos+FPix2_pos', 
                           'FPix1_neg+FPix2_neg',
                                
                           'BPix3+TIB1', 
                           'BPix3+TIB2']
_convLayerPairsLayerList.extend(_convLayerPairsStripOnlyLayers)
                                                        
    

import RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi as _mod

convLayerPairs = _mod.seedingLayersEDProducer.clone(
                                layerList = _convLayerPairsLayerList,
                                BPix = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    HitProducer = cms.string('siPixelRecHits'),
                                    skipClusters = cms.InputTag('convClusters'),
                                    ),
                                FPix = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'),
                                    HitProducer = cms.string('siPixelRecHits'),
                                    skipClusters = cms.InputTag('convClusters'),
                                    ),
                                TIB = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
                                    matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
                                    skipClusters = cms.InputTag('convClusters'),
                                    ),
                                MTIB = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
                                    rphiRecHits = cms.InputTag('siStripMatchedRecHits','rphiRecHit'),
                                    skipClusters = cms.InputTag('convClusters'),
                                    ),
                                TID = cms.PSet(
                                    useSimpleRphiHitsCleaner = cms.bool(False),
                                    matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
                                    useRingSlector = cms.bool(True),
                                    TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
                                    maxRing = cms.int32(2),
                                    minRing = cms.int32(1),
                                    skipClusters = cms.InputTag('convClusters'),
                                    ),
                                TEC = cms.PSet(
                                    useSimpleRphiHitsCleaner = cms.bool(False),
                                    minRing = cms.int32(1),
                                    matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
                                    useRingSlector = cms.bool(True),
                                    TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
                                    rphiRecHits = cms.InputTag('siStripMatchedRecHits','rphiRecHitUnmatched'),
                                    maxRing = cms.int32(7),
                                    stereoRecHits = cms.InputTag('siStripMatchedRecHits','stereoRecHitUnmatched'),
                                    skipClusters = cms.InputTag('convClusters'),
                                    ),
                                TOB = cms.PSet(
                                    matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
                                    TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
                                    skipClusters = cms.InputTag('convClusters'),
                                    ),
                                MTOB = cms.PSet(
                                    TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
                                    rphiRecHits = cms.InputTag('siStripMatchedRecHits','rphiRecHit'),
                                    skipClusters = cms.InputTag('convClusters'),
                                    ),
                                )

#this was done by Sam Harper (Aug 2017) and is a best guess (and I do guess wrong sometimes)
#in general I kept all the old layer pairs and added sensible phaseI combinations
#most pairs are adjacent layers (with some small exceptions) so I stuck to that
_convLayerPairsLayerListPhaseI = ['BPix1+BPix2', 
                                  
                                  'BPix2+BPix3', 
                                  'BPix3+BPix4',#added for PhaseI
                                  
                                  'BPix2+FPix1_pos', 
                                  'BPix2+FPix1_neg', 
                                  'BPix2+FPix2_pos', 
                                  'BPix2+FPix2_neg',                                        
                                  'BPix3+FPix1_pos', #added for phaseI 
                                  'BPix3+FPix1_neg', #added for phaseI 
                                  
                                  'FPix1_pos+FPix2_pos', 
                                  'FPix1_neg+FPix2_neg',
                                  
                                  'FPix2_pos+FPix3_pos', #added for phaseI 
                                  'FPix2_neg+FPix3_neg',#added for phaseI 
                                  
                                  'BPix3+TIB1', 
                                  #'BPix3+TIB2' #removed for phaseI
                                  'BPix4+TIB1', #added for phase I
                                  'BPix4+TIB2', #added for phase I
                                  ] 
_convLayerPairsLayerListPhaseI.extend(_convLayerPairsStripOnlyLayers)

from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(convLayerPairs, layerList = _convLayerPairsLayerListPhaseI)


trackingPhase2PU140.toReplaceWith(convLayerPairs, cms.EDProducer('SeedingLayersEDProducer',
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
)

photonConvTrajSeedFromSingleLeg.TrackRefitter      = 'generalTracks'
photonConvTrajSeedFromSingleLeg.primaryVerticesTag = 'firstStepPrimaryVertices'
#photonConvTrajSeedFromQuadruplets.TrackRefitter      = 'generalTracks'
#photonConvTrajSeedFromQuadruplets.primaryVerticesTag = 'pixelVertices'
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU
trackingLowPU.toModify(photonConvTrajSeedFromSingleLeg, primaryVerticesTag   = 'pixelVertices')
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(photonConvTrajSeedFromSingleLeg, vtxMinDoF = 999999.)
    
# TRACKER DATA CONTROL

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
convCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
        maxLostHits         = 1,
        minimumNumberOfHits = 3,
        minPt               = 0.1
)


import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
convStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName    = 'convStepChi2Est',
    nSigma           = 3.0,
    MaxChi2          = 30.0,
    MaxDisplacement  = 100,
    MaxSagitta       = -1.,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutTight')
)


# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
_convCkfTrajectoryBuilderBase = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = dict(refToPSet_ = 'convCkfTrajectoryFilter'),
    minNrOfHitsForRebuild = 3,
    maxCand = 1
)

convCkfTrajectoryBuilder = _convCkfTrajectoryBuilderBase.clone(
    estimator = 'convStepChi2Est'
)

trackingPhase2PU140.toReplaceWith(convCkfTrajectoryBuilder, _convCkfTrajectoryBuilderBase.clone(
    maxCand = 2
))

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
convTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'photonConvTrajSeedFromSingleLeg:convSeedCandidates',
    clustersToSkip =  cms.InputTag('convClusters'),
    TrajectoryBuilderPSet = dict(refToPSet_ = 'convCkfTrajectoryBuilder')
)

trackingPhase2PU140.toModify(convTrackCandidates,
    clustersToSkip = None,
    phase2clustersToSkip = cms.InputTag('convClusters')
)

import TrackingTools.TrackFitters.RungeKuttaFitters_cff
convStepFitterSmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
    ComponentName = 'convStepFitterSmoother',
    EstimateCut   = 30,
    Smoother      = 'convStepRKSmoother'
)
    
convStepRKTrajectorySmoother = TrackingTools.TrackFitters.RungeKuttaFitters_cff.RKTrajectorySmoother.clone(
    ComponentName  = 'convStepRKSmoother',
    errorRescaling = 10.0
)

        
# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
convStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'convTrackCandidates',
    AlgorithmName = 'conversionStep',
    Fitter = 'convStepFitterSmoother'
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
            d0_par1 = ( 5., 8.0 ), # not sure these values are sane....
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

ConvStepTask = cms.Task( convClusters 
                         , convLayerPairs
                         , photonConvTrajSeedFromSingleLeg 
                         , convTrackCandidates
                         , convStepTracks
                         , convStepSelector
                         #+ Conv2Step #full quad-seeding sequence
                         )
ConvStep = cms.Sequence( ConvStepTask ) 


### Quad-seeding sequence disabled (#+ Conv2Step)
# if enabled, the quad-seeded tracks have to be merged with the single-leg seeded tracks
# in RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff change:
###
#conversionStepTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
#    TrackProducers = 'convStepTracks',
#    hasSelector=1,
#    selectedTrackQuals = 'convStepSelector:convStep',
#    setsToMerge = dict( tLists = 1, pQual = True ),
#    copyExtras = True,
#    makeReKeyedSeeds = False
#    )
###
# TO this:
###
#conversionStepTracks = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
#    TrackProducers = ['convStepTracks', 'conv2StepTracks'],
#    hasSelector = [1,1],
#    selectedTrackQuals = ['convStepSelector:convStep', 'conv2StepSelector:conv2Step'],
#    setsToMerge = dict( tLists = [0,1], pQual = True ),
#    copyExtras = True,
#    makeReKeyedSeeds = False
#    )
###
