import FWCore.ParameterSet.Config as cms

# This step runs over all clusters

# run only if there are high pT jets
from RecoJets.JetProducers.TracksForJets_cff import trackRefsForJets
initialStepTrackRefsForJets = trackRefsForJets.clone(src = cms.InputTag('initialStepTracks'))
from RecoJets.JetProducers.caloJetsForTrk_cff import *
jetsForCoreTracking = cms.EDFilter("CandPtrSelector", src = cms.InputTag("ak4CaloJetsForTrk"), cut = cms.string("pt > 100 && abs(eta) < 2.5"))

# care only at tracks from main PV
firstStepGoodPrimaryVertices = cms.EDFilter("PrimaryVertexObjectFilter",
     filterParams = cms.PSet(
     	     minNdof = cms.double(25.0),
             maxZ = cms.double(15.0),
             maxRho = cms.double(2.0)
     ),
     src=cms.InputTag('firstStepPrimaryVertices')
)

# SEEDING LAYERS
jetCoreRegionalStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 
                            'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
                            'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
                            'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg',
                            #'BPix2+TIB1','BPix2+TIB2',
                            'BPix3+TIB1','BPix3+TIB2'),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        hitErrorRZ = cms.double(0.006),
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        #skipClusters = cms.InputTag('jetCoreRegionalStepClusters')
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        #skipClusters = cms.InputTag('jetCoreRegionalStepClusters')
    )
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff
jetCoreRegionalStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
jetCoreRegionalStepSeeds.RegionFactoryPSet = cms.PSet(
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),#not so nice to depend on RecoTau...
      RegionPSet = cms.PSet(
        precise = cms.bool( True ),
        originRadius = cms.double( 0.2 ),
        ptMin = cms.double( 10. ),
        originHalfLength = cms.double( 0.2 ),
        deltaPhiRegion = cms.double( 0.20 ), 
        deltaEtaRegion = cms.double( 0.20 ), 
        JetSrc = cms.InputTag( "jetsForCoreTracking" ),
#       JetSrc = cms.InputTag( "ak5CaloJets" ),
        vertexSrc = cms.InputTag( "firstStepGoodPrimaryVertices" ),
        measurementTrackerName = cms.string( "MeasurementTrackerEvent" ),
        howToUseMeasurementTracker = cms.double( -1.0 )
      )
)
jetCoreRegionalStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'jetCoreRegionalStepSeedLayers'
jetCoreRegionalStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none'),
#PixelClusterShapeSeedComparitor'),
#        FilterAtHelixStage = cms.bool(True),
#        FilterPixelHits = cms.bool(True),
#        FilterStripHits = cms.bool(False),
#       ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
    )
jetCoreRegionalStepSeeds.SeedCreatorPSet.forceKinematicWithRegionDirection = cms.bool( True )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
jetCoreRegionalStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.1
)

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi
jetCoreRegionalStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = cms.string('jetCoreRegionalStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
jetCoreRegionalStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('jetCoreRegionalStepTrajectoryFilter')),
    #clustersToSkip = cms.InputTag('jetCoreRegionalStepClusters'),
    maxCand = 50,
    estimator = cms.string('jetCoreRegionalStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
jetCoreRegionalStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('jetCoreRegionalStepSeeds'),
    maxSeedsBeforeCleaning = cms.uint32(10000),
    TrajectoryBuilderPSet = cms.PSet( refToPSet_ = cms.string('jetCoreRegionalStepTrajectoryBuilder')),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    #numHitsForSeedCleaner = cms.int32(50),
    #onlyPixelHitsForSeedCleaner = cms.bool(True),
)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
jetCoreRegionalStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('jetCoreRegionalStep'),
    src = 'jetCoreRegionalStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

# Final selection
import RecoTracker.IterativeTracking.LowPtTripletStep_cff
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
jetCoreRegionalStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src='jetCoreRegionalStepTracks',
    trackSelectors= cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'jetCoreRegionalStepLoose',
            ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'jetCoreRegionalStepTight',
            preFilterName = 'jetCoreRegionalStepLoose',
            ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'jetCoreRegionalStep',
            preFilterName = 'jetCoreRegionalStepTight',
            ),
        ) #end of vpset
    ) #end of clone

# Final sequence
JetCoreRegionalStep = cms.Sequence(initialStepTrackRefsForJets*caloJetsForTrk*jetsForCoreTracking*
                                   firstStepGoodPrimaryVertices*
                                   #jetCoreRegionalStepClusters*
                                   jetCoreRegionalStepSeedLayers*
                                   jetCoreRegionalStepSeeds*
                                   jetCoreRegionalStepTrackCandidates*
                                   jetCoreRegionalStepTracks*
                                   jetCoreRegionalStepSelector)
