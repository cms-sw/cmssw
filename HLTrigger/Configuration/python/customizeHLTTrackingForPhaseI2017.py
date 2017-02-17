import FWCore.ParameterSet.Config as cms

# customisation functions for the HLT configuration
from HLTrigger.Configuration.common import *

# import the relevant eras from Configuration.Eras.*
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel


# modify the HLT configuration for the Phase I pixel geometry
def customizeHLTPhaseIPixelGeom(process):

    for esproducer in esproducers_by_type(process,"ClusterShapeHitFilterESProducer"):
         esproducer.PixelShapeFile = 'RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1TkNewFPix.par'
    for producer in producers_by_type(process,"SiPixelRawToDigi"):
        if "hlt" in producer.label():
            producer.UsePhase1 = cms.bool( True )
    return process

# attach `modifyHLTPhaseIPixelGeom' to the `phase1Pixel` era
def modifyHLTPhaseIPixelGeom(process):
    phase1Pixel.toModify(process, customizeHLTPhaseIPixelGeom)



# modify the HLT configuration to run the Phase I tracking in the particle flow sequence
def customizeHLTForPFTrackingPhaseI2017(process):

    process.hltPixelLayerTriplets.layerList = cms.vstring(
        'BPix1+BPix2+BPix3',
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
        'FPix1_neg+FPix2_neg+FPix3_neg' 
    )
    process.hltPixelLayerQuadruplets = cms.EDProducer("SeedingLayersEDProducer",
         BPix = cms.PSet(
            useErrorsFromParam = cms.bool( True ),
            hitErrorRPhi = cms.double( 0.0027 ),
            TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
            HitProducer = cms.string( "hltSiPixelRecHits" ),
            hitErrorRZ = cms.double( 0.006 )
        ),
        FPix = cms.PSet(
            useErrorsFromParam = cms.bool( True ),
            hitErrorRPhi = cms.double( 0.0051 ),
            TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
            HitProducer = cms.string( "hltSiPixelRecHits" ),
            hitErrorRZ = cms.double( 0.0036 )
        ),
        MTEC = cms.PSet( ),
        MTIB = cms.PSet( ),
        MTID = cms.PSet( ),
        MTOB = cms.PSet( ),
        TEC = cms.PSet( ),
        TIB = cms.PSet( ),
        TID = cms.PSet( ),
        TOB = cms.PSet( ),
        layerList = cms.vstring(
            'BPix1+BPix2+BPix3+BPix4', 
            'BPix1+BPix2+BPix3+FPix1_pos', 
            'BPix1+BPix2+BPix3+FPix1_neg', 
            'BPix1+BPix2+FPix1_pos+FPix2_pos', 
            'BPix1+BPix2+FPix1_neg+FPix2_neg', 
            'BPix1+FPix1_pos+FPix2_pos+FPix3_pos', 
            'BPix1+FPix1_neg+FPix2_neg+FPix3_neg'
        )
    )

    # Configure seed generator / pixel track producer
    from RecoPixelVertexing.PixelTriplets.caHitQuadrupletEDProducer_cfi import caHitQuadrupletEDProducer as _caHitQuadrupletEDProducer

    process.hltPixelTracksTrackingRegions.RegionPSet = cms.PSet(
        precise = cms.bool( True ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        originRadius = cms.double(0.02),
        ptMin = cms.double(0.8),
        nSigmaZ = cms.double(4.0),
    )

    process.hltPixelTracksHitDoublets.seedingLayers = "hltPixelLayerQuadruplets"
    process.hltPixelTracksHitDoublets.layerPairs = [0,1,2] # layer pairs (0,1), (1,2), (2,3)

    process.hltPixelTracksHitQuadruplets = _caHitQuadrupletEDProducer.clone(
        doublets = "hltPixelTracksHitDoublets",
        extraHitRPhitolerance = cms.double(0.032),
        maxChi2 = dict(
            pt1    = 0.7,
            pt2    = 2,
            value1 = 200,
            value2 = 50,
            enabled = True,
        ),
        useBendingCorrection = True,
        fitFastCircle = True,
        fitFastCircleChi2Cut = True,
        CAThetaCut = cms.double(0.002),
        CAPhiCut = cms.double(0.2),
        CAHardPtCut = cms.double(0),
        SeedComparitorPSet = cms.PSet( 
            ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
            clusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersCache" )
        )
    )

    process.hltPixelTracks.SeedingHitSets = "hltPixelTracksHitQuadruplets"

    process.HLTDoRecoPixelTracksSequence = cms.Sequence(
        process.hltPixelLayerQuadruplets +
        process.hltPixelTracksTrackingRegions +
        process.hltPixelTracksHitDoublets +
        process.hltPixelTracksHitQuadruplets +
        process.hltPixelTracks
    )
    
    process.HLTIter0PSetTrajectoryFilterIT.minimumNumberOfHits = cms.int32( 4 )
    process.HLTIter0PSetTrajectoryFilterIT.minHitsMinPt        = cms.int32( 4 )
    process.hltIter0PFlowTrackCutClassifier.mva.minLayers    = cms.vint32( 3, 3, 4 )
    process.hltIter0PFlowTrackCutClassifier.mva.min3DLayers  = cms.vint32( 0, 3, 4 )
    process.hltIter0PFlowTrackCutClassifier.mva.minPixelHits = cms.vint32( 0, 3, 4 )

    process.HLTIter0GroupedCkfTrajectoryBuilderIT = cms.PSet(
	        ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
	        bestHitOnly = cms.bool(True),
        	propagatorAlong = cms.string('PropagatorWithMaterialParabolicMf'),
        	trajectoryFilter = cms.PSet(refToPSet_ = cms.string('HLTIter0PSetTrajectoryFilterIT')),
        	inOutTrajectoryFilter = cms.PSet(refToPSet_ = cms.string('HLTIter0PSetTrajectoryFilterIT')),
	        useSameTrajFilter = cms.bool(True),
	        maxCand = cms.int32(2),
	        intermediateCleaning = cms.bool(True),
	        lostHitPenalty = cms.double(30.0),
	        MeasurementTrackerName = cms.string('hltESPMeasurementTracker'),
	        lockHits = cms.bool(True),
	        TTRHBuilder = cms.string('hltESPTTRHBWithTrackAngle'),
	        foundHitBonus = cms.double(5.0),
	        updator = cms.string('hltESPKFUpdator'),
	        alwaysUseInvalidHits = cms.bool(False),
	        requireSeedHitsInRebuild = cms.bool(True),
	        keepOriginalIfRebuildFails = cms.bool(False),
	        estimator = cms.string('hltESPChi2ChargeMeasurementEstimator9'),
     		propagatorOpposite = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
	        minNrOfHitsForRebuild = cms.int32(5),
	        maxDPhiForLooperReconstruction = cms.double(2.0),
	        maxPtForLooperReconstruction = cms.double(0.7),
	        cleanTrajectoryAfterInOut = cms.bool( False ),
	        useHitsSplitting = cms.bool( False ),
	        doSeedingRegionRebuilding = cms.bool( False )
    )

    process.hltIter0PFlowCkfTrackCandidates.TrajectoryBuilder = cms.string('HLTIter0GroupedCkfTrajectoryBuilderIT')
    process.hltIter0PFlowCkfTrackCandidates.TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('HLTIter0GroupedCkfTrajectoryBuilderIT'))


    process.hltIter1PixelLayerQuadruplets = cms.EDProducer("SeedingLayersEDProducer",
        BPix = cms.PSet(
          useErrorsFromParam = cms.bool( True ),
          hitErrorRPhi = cms.double( 0.0027 ),
          TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
          HitProducer = cms.string( "hltSiPixelRecHits" ),
          hitErrorRZ = cms.double( 0.006 ),
          skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
        ),
        FPix = cms.PSet(
          useErrorsFromParam = cms.bool( True ),
          hitErrorRPhi = cms.double( 0.0051 ),
          TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
          HitProducer = cms.string( "hltSiPixelRecHits" ),
          hitErrorRZ = cms.double( 0.0036 ),
          skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
        ),
        MTEC = cms.PSet( ),
        MTIB = cms.PSet( ),
        MTID = cms.PSet( ),
        MTOB = cms.PSet( ),
        TEC = cms.PSet( ),
        TIB = cms.PSet( ),
        TID = cms.PSet( ),
        TOB = cms.PSet( ),
        layerList = cms.vstring(
            'BPix1+BPix2+BPix3+BPix4', 
            'BPix1+BPix2+BPix3+FPix1_pos', 
            'BPix1+BPix2+BPix3+FPix1_neg', 
            'BPix1+BPix2+FPix1_pos+FPix2_pos', 
            'BPix1+BPix2+FPix1_neg+FPix2_neg', 
            'BPix1+FPix1_pos+FPix2_pos+FPix3_pos', 
            'BPix1+FPix1_neg+FPix2_neg+FPix3_neg'
        )
    )




    process.hltIter1PixelTracks = process.hltPixelTracks.clone()

    process.hltIter1PixelTracks.SeedingHitSets = "hltIter1PFlowPixelHitQuadruplets"
    process.hltIter1PFlowPixelHitDoublets.layerPairs = [0,1,2] # layer pairs (0,1), (1,2), (2,3)
    process.hltIter1PFlowPixelHitDoublets.seedingLayers = "hltIter1PixelLayerQuadruplets"


    from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
    process.hltIter1PFlowPixelTrackingRegions = _globalTrackingRegionWithVertices.clone() 
    process.hltIter1PFlowPixelTrackingRegions.RegionPSet.sigmaZVertex = cms.double(4.0)
    process.hltIter1PFlowPixelTrackingRegions.RegionPSet.nSigmaZ = cms.double(4.0)
    process.hltIter1PFlowPixelTrackingRegions.RegionPSet.originRadius = cms.double(0.05)
    process.hltIter1PFlowPixelTrackingRegions.RegionPSet.ptMin = cms.double(0.3)
    process.hltIter1PFlowPixelTrackingRegions.RegionPSet.fixedError = cms.double(0.2)
    process.hltIter1PFlowPixelTrackingRegions.RegionPSet.useFakeVertices = cms.bool(True)
    process.hltIter1PFlowPixelTrackingRegions.RegionPSet.useMultipleScattering = cms.bool(True)
    process.hltIter1PFlowPixelTrackingRegions.RegionPSet.beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
    process.hltIter1PFlowPixelTrackingRegions.RegionPSet.VertexCollection = cms.InputTag( "hltTrimmedPixelVertices" )

    process.hltIter1PFlowPixelHitQuadruplets = _caHitQuadrupletEDProducer.clone(
        doublets = "hltIter1PFlowPixelHitDoublets",
        extraHitRPhitolerance = cms.double(0.032),
        maxChi2 = dict(
            pt1    = 0.8,
            pt2    = 2,
            value1 = 2000,
            value2 = 100,
            enabled = True,
        ),
        useBendingCorrection = True,
        fitFastCircle = True,
        fitFastCircleChi2Cut = True,
        CAThetaCut = cms.double(0.004),
        CAPhiCut = cms.double(0.1),
        CAHardPtCut = cms.double(0),
        SeedComparitorPSet = cms.PSet( 
            ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
            clusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersCache" )
        )


    )


 
    process.hltIter1PFlowPixelSeeds = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
        useEventsWithNoVertex = cms.bool( True ),
        originHalfLength = cms.double( 0.3 ), 
        useProtoTrackKinematics = cms.bool( False ),
        usePV = cms.bool( True ),
        SeedCreatorPSet = cms.PSet(  refToPSet_ = cms.string( "HLTSeedFromProtoTracks" ) ),
        InputVertexCollection = cms.InputTag( "hltTrimmedPixelVertices" ),
        TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
        InputCollection = cms.InputTag( "hltIter1PixelTracks" ),
        originRadius = cms.double( 0.1 )
    )
    process.HLTIter1PSetTrajectoryBuilderIT.strictSeedExtension = cms.bool(True)
  

    process.HLTIter1PSetTrajectoryFilterIT = cms.PSet( 
	    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
	    chargeSignificance = cms.double(-1.0),
	    constantValueForLostHitsFractionFilter = cms.double(2.0),
	    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
	    maxCCCLostHits = cms.int32(0),
	    maxConsecLostHits = cms.int32(1),
	    maxLostHits = cms.int32(1),  
	    maxLostHitsFraction = cms.double(0.1),
	    maxNumberOfHits = cms.int32(100),
	    minGoodStripCharge = cms.PSet( refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
	    minHitsMinPt = cms.int32(3),
	     minNumberOfHitsForLoopers = cms.int32(13),
	    minNumberOfHitsPerLoop = cms.int32(4),
	    minPt = cms.double(0.2),
	    minimumNumberOfHits = cms.int32(4), 
	    nSigmaMinPt = cms.double(5.0),
	    pixelSeedExtension = cms.bool(True),
	    seedExtension = cms.int32(1),
	    seedPairPenalty = cms.int32(0),
	    strictSeedExtension = cms.bool(True)
    )

    process.HLTIter1PSetTrajectoryFilterInOutIT = cms.PSet(
	    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
	    chargeSignificance = cms.double(-1.0),
	    constantValueForLostHitsFractionFilter = cms.double(2.0),
	    extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
	    maxCCCLostHits = cms.int32(0),
	    maxConsecLostHits = cms.int32(1),
	    maxLostHits = cms.int32(1),  
	    maxLostHitsFraction = cms.double(0.1),
	    maxNumberOfHits = cms.int32(100),
	    minGoodStripCharge = cms.PSet( refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
	    minHitsMinPt = cms.int32(3),
	    minNumberOfHitsForLoopers = cms.int32(13),
	    minNumberOfHitsPerLoop = cms.int32(4),
	    minPt = cms.double(0.2),
	    minimumNumberOfHits = cms.int32(4),
	    nSigmaMinPt = cms.double(5.0),
	    pixelSeedExtension = cms.bool(True),
	    seedExtension = cms.int32(1),
	    seedPairPenalty = cms.int32(0),
	    strictSeedExtension = cms.bool(True)
    )

    process.HLTIter1PSetTrajectoryBuilderIT = cms.PSet( 
	  inOutTrajectoryFilter = cms.PSet( refToPSet_ = cms.string('HLTIter1PSetTrajectoryFilterInOutIT') ),
	  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
	  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter1PSetTrajectoryFilterIT" ) ),
	  maxCand = cms.int32( 2 ),
	  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
	  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
	  MeasurementTrackerName = cms.string( "hltIter1ESPMeasurementTracker" ),
	  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
	  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	  updator = cms.string( "hltESPKFUpdator" ),
	  alwaysUseInvalidHits = cms.bool( False ),
	  intermediateCleaning = cms.bool( True ),
	  lostHitPenalty = cms.double( 30.0 ),
	  useSameTrajFilter = cms.bool(False) 
    )
    process.HLTIter1GroupedCkfTrajectoryBuilderIT = cms.PSet(
        	ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
	        bestHitOnly = cms.bool(True),
	        propagatorAlong = cms.string('PropagatorWithMaterialParabolicMf'),
	        trajectoryFilter = cms.PSet(refToPSet_ = cms.string('HLTIter1PSetTrajectoryFilterIT')),
	        inOutTrajectoryFilter = cms.PSet(refToPSet_ = cms.string('HLTIter1PSetTrajectoryFilterInOutIT')),
	        useSameTrajFilter = cms.bool(False),
	        maxCand = cms.int32(2),
	        intermediateCleaning = cms.bool(True),
	        lostHitPenalty = cms.double(30.0),
	        MeasurementTrackerName = cms.string('hltIter1ESPMeasurementTracker'),
	        lockHits = cms.bool(True),
	        TTRHBuilder = cms.string('hltESPTTRHBWithTrackAngle'),
	        foundHitBonus = cms.double(5.0),
	        updator = cms.string('hltESPKFUpdator'),
	        alwaysUseInvalidHits = cms.bool(False),
	        requireSeedHitsInRebuild = cms.bool(True),
	        keepOriginalIfRebuildFails = cms.bool(False),
	        estimator = cms.string('hltESPChi2ChargeMeasurementEstimator16'),
	        propagatorOpposite = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
	        minNrOfHitsForRebuild = cms.int32(5)
    )

    process.hltIter1PFlowCkfTrackCandidates.TrajectoryBuilder = cms.string('HLTIter1GroupedCkfTrajectoryBuilderIT')
    process.hltIter1PFlowCkfTrackCandidates.TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('HLTIter1GroupedCkfTrajectoryBuilderIT'))


    process.HLTIterativeTrackingIteration1 = cms.Sequence( process.hltIter1ClustersRefRemoval + process.hltIter1MaskedMeasurementTrackerEvent + process.hltIter1PixelLayerQuadruplets + process.hltIter1PFlowPixelTrackingRegions + process.hltIter1PFlowPixelClusterCheck + process.hltIter1PFlowPixelHitDoublets + process.hltIter1PFlowPixelHitQuadruplets + process.hltIter1PFlowPixelSeeds + process.hltIter1PFlowCkfTrackCandidates + process.hltIter1PFlowCtfWithMaterialTracks + process.hltIter1PFlowTrackCutClassifierPrompt + process.hltIter1PFlowTrackCutClassifierDetached + process.hltIter1PFlowTrackCutClassifierMerged + process.hltIter1PFlowTrackSelectionHighPurity )

    process.hltIter2PixelLayerTriplets = cms.EDProducer( "SeedingLayersEDProducer",
        layerList = cms.vstring( 
	      	'BPix1+BPix2+BPix3',
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
      		'FPix1_neg+FPix2_neg+FPix3_neg' 
        ),
        MTOB = cms.PSet( ),
        TEC = cms.PSet( ),
        MTID = cms.PSet( ),
        FPix = cms.PSet( 
            HitProducer = cms.string( "hltSiPixelRecHits" ),
            hitErrorRZ = cms.double( 0.0036 ),
            useErrorsFromParam = cms.bool( True ),
            TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
            skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" ),
            hitErrorRPhi = cms.double( 0.0051 )
        ),
        MTEC = cms.PSet( ),
        MTIB = cms.PSet( ),
        TID = cms.PSet( ),
        TOB = cms.PSet( ),
        BPix = cms.PSet( 
            HitProducer = cms.string( "hltSiPixelRecHits" ),
            hitErrorRZ = cms.double( 0.006 ),
            useErrorsFromParam = cms.bool( True ),
            TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
            skipClusters = cms.InputTag( "hltIter2ClustersRefRemoval" ),
            hitErrorRPhi = cms.double( 0.0027 )
        ),
        TIB = cms.PSet( )
    )

 
    process.hltIter2PFlowPixelTrackingRegions.RegionPSet.ptMin = cms.double(0.8)
    process.hltIter2PFlowPixelTrackingRegions.RegionPSet.originRadius = cms.double(0.025)
    process.hltIter2PFlowPixelTrackingRegions.RegionPSet.nSigmaZVertex = cms.double(4.0)

    from RecoPixelVertexing.PixelTriplets.caHitTripletEDProducer_cfi import caHitTripletEDProducer as _caHitTripletEDProducer

    process.hltIter2PFlowPixelHitDoublets.seedingLayers = "hltIter2PixelLayerTriplets"
    process.hltIter2PFlowPixelHitDoublets.produceIntermediateHitDoublets = True
    process.hltIter2PFlowPixelHitDoublets.produceSeedingHitSets = False
    process.hltIter2PFlowPixelHitDoublets.layerPairs = [0,1]
    process.hltIter2PFlowPixelHitTriplets = _caHitTripletEDProducer.clone(
        doublets = cms.InputTag("hltIter2PFlowPixelHitDoublets"),
        extraHitRPhitolerance = cms.double(0.032),
    	maxChi2 = cms.PSet(
        	pt1    = cms.double(0.8), pt2    = cms.double(8),
        	value1 = cms.double(100), value2 = cms.double(6),
        	enabled = cms.bool(True),
    	),
    	useBendingCorrection = cms.bool(True),
    	CAThetaCut = cms.double(0.004),
    	CAPhiCut = cms.double(0.1),
    	CAHardPtCut = cms.double(0.3),

    )

    def _copy(old, new, skip=[]):
        skipSet = set(skip)
        for key in old.parameterNames_():
            if key not in skipSet:
                setattr(new, key, getattr(old, key))
    from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cfi import seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer as _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer
    process.hltIter2PFlowPixelSeeds = _seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer.clone(seedingHitSets="hltIter2PFlowPixelHitTriplets")
    _copy(process.HLTSeedFromConsecutiveHitsTripletOnlyCreator, process.hltIter2PFlowPixelSeeds, skip=["ComponentName"])

    process.HLTIter2GroupedCkfTrajectoryBuilderIT = cms.PSet(

        	ComponentType = cms.string('GroupedCkfTrajectoryBuilder'),
	        bestHitOnly = cms.bool(True),
	        propagatorAlong = cms.string('PropagatorWithMaterialParabolicMf'),
	        trajectoryFilter = cms.PSet(refToPSet_ = cms.string('HLTIter2PSetTrajectoryFilterIT')),
	        inOutTrajectoryFilter = cms.PSet(refToPSet_ = cms.string('HLTIter2PSetTrajectoryFilterIT')),
	        useSameTrajFilter = cms.bool(True),
	        maxCand = cms.int32(2),
	        intermediateCleaning = cms.bool(True),
	        lostHitPenalty = cms.double(30.0),
	        MeasurementTrackerName = cms.string('hltESPMeasurementTracker'),
	        lockHits = cms.bool(True),
	        TTRHBuilder = cms.string('hltESPTTRHBWithTrackAngle'),
	        foundHitBonus = cms.double(5.0),
	        updator = cms.string('hltESPKFUpdator'),
	        alwaysUseInvalidHits = cms.bool(False),
	        requireSeedHitsInRebuild = cms.bool(True),
	        keepOriginalIfRebuildFails = cms.bool(False),
	        estimator = cms.string('hltESPChi2ChargeMeasurementEstimator16'),
	        propagatorOpposite = cms.string('PropagatorWithMaterialParabolicMfOpposite'),
	        minNrOfHitsForRebuild = cms.int32(5),
	        maxDPhiForLooperReconstruction = cms.double(2.0),
	        maxPtForLooperReconstruction = cms.double(0.7),
	        cleanTrajectoryAfterInOut = cms.bool( False ),
	        useHitsSplitting = cms.bool( False ),
	        doSeedingRegionRebuilding = cms.bool( False )
    )	

    process.hltIter2PFlowCkfTrackCandidates.TrajectoryBuilder = cms.string('HLTIter2GroupedCkfTrajectoryBuilderIT')
    process.hltIter2PFlowCkfTrackCandidates.TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('HLTIter2GroupedCkfTrajectoryBuilderIT'))



    process.HLTIterativeTrackingIteration2 = cms.Sequence( process.hltIter2ClustersRefRemoval + process.hltIter2MaskedMeasurementTrackerEvent + process.hltIter2PixelLayerTriplets + process.hltIter2PFlowPixelTrackingRegions + process.hltIter2PFlowPixelClusterCheck + process.hltIter2PFlowPixelHitDoublets + process.hltIter2PFlowPixelHitTriplets + process.hltIter2PFlowPixelSeeds + process.hltIter2PFlowCkfTrackCandidates + process.hltIter2PFlowCtfWithMaterialTracks + process.hltIter2PFlowTrackCutClassifier + process.hltIter2PFlowTrackSelectionHighPurity )

    # Need to operate on Paths as well...
    for seqs in [process.sequences_(), process.paths_()]:
        for seqName, seq in seqs.iteritems():
            from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor
            l = list()
            v = ModuleNodeVisitor(l)
            seq.visit(v)

            if process.hltPixelTracks in l and not process.hltPixelLayerQuadruplets in l:
                seq.remove(process.hltPixelLayerTriplets) # note that this module does not necessarily exist in sequence 'seq', if it doesn't, it does not get removed
                index = seq.index(process.hltPixelTracksHitDoublets)
                seq.insert(index,process.hltPixelLayerQuadruplets)
                index = seq.index(process.hltPixelTracksHitTriplets)
                seq.remove(process.hltPixelTracksHitTriplets)
                seq.insert(index, process.hltPixelTracksHitQuadruplets)

    # Remove entirely to avoid warning from the early deleter
    del process.hltPixelTracksHitTriplets

    for producer in producers_by_type(process,"PixelTripletHLTEDProducer"):
        if "hltIter1PFlowPixelHitTriplets" in producer.label():

    	   	for seqs in [process.sequences_(), process.paths_()]:
        		for seqName, seq in seqs.iteritems():
            			from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor
            			l = list()
            			v = ModuleNodeVisitor(l)
            			seq.visit(v)

            			if process.hltIter1PFlowPixelHitTriplets in l and not process.hltIter1PFlowPixelHitQuadruplets in l:
                			index = seq.index(process.hltIter1PFlowPixelHitTriplets)
                			seq.insert(index, process.hltIter1PixelTracks)
                			seq.insert(index, process.hltIter1PFlowPixelHitQuadruplets)
                			seq.remove(process.hltIter1PFlowPixelHitTriplets)

    # Remove entirely to avoid warning from the early deleter
    del process.hltIter1PFlowPixelHitTriplets





    return process

# attach `customizeHLTForPFTrackingPhaseI2017` to the `phase1Pixel` era
def modifyHLTForPFTrackingPhaseI2017(process):
    phase1Pixel.toModify(process, customizeHLTForPFTrackingPhaseI2017)
