import FWCore.ParameterSet.Config as cms

# This step runs over all clusters

# run only if there are high pT jets
from RecoJets.JetProducers.TracksForJets_cff import trackRefsForJets
hiInitialStepTrackRefsForJets = trackRefsForJets.clone(src = 'hiGlobalPrimTracks')

#change this to import Bkg substracted Heavy Ion jets:
from RecoHI.HiJetAlgos.hiCaloJetsForTrk_cff import *

hiJetsForCoreTracking = cms.EDFilter("CandPtrSelector", src = cms.InputTag("akPu4CaloJetsSelected"), cut = cms.string("pt > 30 && abs(eta) < 2.5"))

# care only at tracks from main PV
hiFirstStepGoodPrimaryVertices = cms.EDFilter("PrimaryVertexObjectFilter",
     filterParams = cms.PSet(
     	     minNdof = cms.double(25.0),
             maxZ = cms.double(15.0),
             maxRho = cms.double(2.0)
     ),
     src=cms.InputTag('hiSelectedPixelVertex')
)

# SEEDING LAYERS
hiJetCoreRegionalStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
    'BPix1+BPix2+FPix1_pos', 
    'BPix1+BPix2+FPix1_neg', 
    'BPix1+FPix1_pos+FPix2_pos', 
    'BPix1+FPix1_neg+FPix2_neg',
    'BPix1+BPix2+TIB1', 
    'BPix1+BPix3+TIB1', 
    'BPix2+BPix3+TIB1', 
),
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
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
    )
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(hiJetCoreRegionalStepSeedLayers, layerList = ['BPix1+BPix2+BPix3',
    'BPix2+BPix3+BPix4',
    'BPix1+BPix3+BPix4',
    'BPix1+BPix2+BPix4',
    'BPix2+BPix3+FPix1_pos',
    'BPix2+BPix3+FPix1_neg',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix2+FPix1_pos+FPix2_pos',
    'BPix2+FPix1_neg+FPix2_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg',
    'FPix1_pos+FPix2_pos+FPix3_pos',
    'FPix1_neg+FPix2_neg+FPix3_neg',#up to here, same as what is in RecoTracker/TkSeedingLayers/python/PixelLayerTriplets_cfi.py for phase 1
    'BPix1+BPix2+TIB1',#use TIB1 to try to recover tracks w/ 2 hits missing in pix barrel
    'BPix1+BPix3+TIB1',
    'BPix1+BPix4+TIB1',
    'BPix2+BPix3+TIB1',
    'BPix2+BPix4+TIB1',
    'BPix3+BPix4+TIB1',
    ]
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
hiJetCoreRegionalStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = cms.PSet(
	ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),#not so nice to depend on RecoTau...
	RegionPSet = cms.PSet(
	   precise = cms.bool( True ),
	   useMultipleScattering = cms.bool(False),
	   useFakeVertices       = cms.bool(False),
	   originRadius = cms.double( 0.2 ),
	   ptMin = cms.double( 15. ),
	   originHalfLength = cms.double( 0.2 ),
	   deltaPhiRegion = cms.double( 0.30 ), 
	   deltaEtaRegion = cms.double( 0.30 ), 
	   JetSrc = cms.InputTag( "hiJetsForCoreTracking" ),
	   vertexSrc = cms.InputTag( "hiFirstStepGoodPrimaryVertices" ),
	   measurementTrackerName = cms.InputTag( "MeasurementTrackerEvent" ),
	   howToUseMeasurementTracker = cms.string( "Never" )
	)
    ),
    OrderedHitsFactoryPSet = dict(SeedingLayers = 'hiJetCoreRegionalStepSeedLayers'),
    SeedComparitorPSet     = dict(ComponentName = 'none'),
    SeedCreatorPSet        = dict(forceKinematicWithRegionDirection = True),
    ClusterCheckPSet       = dict(doClusterCheck = False)
)
# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
hiJetCoreRegionalStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 6,
    minPt = 10.0
)

import TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi
hiJetCoreRegionalStepChi2Est = TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi.Chi2MeasurementEstimator.clone(
    ComponentName = 'hiJetCoreRegionalStepChi2Est',
    nSigma  = 3.0,
    MaxChi2 = 30.0
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
hiJetCoreRegionalStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = dict(refToPSet_ = 'hiJetCoreRegionalStepTrajectoryFilter'),
    maxCand = 50,
    estimator = 'hiJetCoreRegionalStepChi2Est',
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
hiJetCoreRegionalStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = 'hiJetCoreRegionalStepSeeds',
    maxSeedsBeforeCleaning = 10000,
    TrajectoryBuilderPSet = dict( refToPSet_ = 'hiJetCoreRegionalStepTrajectoryBuilder'),
    NavigationSchool = 'SimpleNavigationSchool',
)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
hiJetCoreRegionalStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = 'jetCoreRegionalStep',
    src = 'hiJetCoreRegionalStepTrackCandidates',
    Fitter = 'FlexibleKFFittingSmoother'
)

# Final selection
import RecoHI.HiTracking.hiMultiTrackSelector_cfi
hiJetCoreRegionalStepSelector = RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiMultiTrackSelector.clone(
    src = 'hiJetCoreRegionalStepTracks',
    vertices = "hiFirstStepGoodPrimaryVertices",
    trackSelectors= cms.VPSet(
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiLooseMTS.clone(
            name = 'hiJetCoreRegionalStepLoose',
            ), #end of pset
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiTightMTS.clone(
            name = 'hiJetCoreRegionalStepTight',
            preFilterName = 'hiJetCoreRegionalStepLoose',
            ),
        RecoHI.HiTracking.hiMultiTrackSelector_cfi.hiHighpurityMTS.clone(
            name = 'hiJetCoreRegionalStep',
            preFilterName = 'hiJetCoreRegionalStepTight',
            min_nhits = 14
            ),
        ) #end of vpset
    ) #end of clone

# Final sequence
hiJetCoreRegionalStepTask = cms.Task(
                                   hiCaloJetsForTrkTask,hiJetsForCoreTracking,
                                   hiFirstStepGoodPrimaryVertices,
                                   hiJetCoreRegionalStepSeedLayers,
                                   hiJetCoreRegionalStepSeeds,
                                   hiJetCoreRegionalStepTrackCandidates,
                                   hiJetCoreRegionalStepTracks,
                                   hiJetCoreRegionalStepSelector)
hiJetCoreRegionalStep = cms.Sequence(hiJetCoreRegionalStepTask)
