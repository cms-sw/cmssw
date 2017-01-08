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
    from RecoPixelVertexing.PixelTriplets.CAHitQuadrupletGenerator_cfi import CAHitQuadrupletGenerator as _CAHitQuadrupletGenerator

    process.hltPixelTracks.OrderedHitsFactoryPSet  = _CAHitQuadrupletGenerator.clone(
        ComponentName = cms.string("CAHitQuadrupletGenerator"),
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
        SeedingLayers = cms.InputTag("hltPixelLayerQuadruplets"),
        CAThetaCut = cms.double(0.0012),
        CAPhiCut = cms.double(0.2),
        CAHardPtCut = cms.double(0),
        SeedComparitorPSet = cms.PSet( 
            ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
            clusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersCache" )
        )
    )

    process.hltPixelTracks.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool( True ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        originRadius = cms.double(0.02),
        ptMin = cms.double(0.9),
        nSigmaZ = cms.double(4.0),
    )

    process.HLTDoRecoPixelTracksSequence = cms.Sequence( process.hltPixelLayerQuadruplets + process.hltPixelTracks )
    
    process.HLTIter0PSetTrajectoryFilterIT.minimumNumberOfHits = cms.int32( 4 )
    process.HLTIter0PSetTrajectoryFilterIT.minHitsMinPt        = cms.int32( 4 )
    process.hltIter0PFlowTrackCutClassifier.mva.minLayers    = cms.vint32( 3, 3, 4 )
    process.hltIter0PFlowTrackCutClassifier.mva.min3DLayers  = cms.vint32( 0, 3, 4 )
    process.hltIter0PFlowTrackCutClassifier.mva.minPixelHits = cms.vint32( 0, 3, 4 )

    process.hltIter1PixelLayerTriplets = cms.EDProducer( "SeedingLayersEDProducer",
        layerList = cms.vstring( 
            'BPix1+BPix2+BPix3', 
            'BPix1+BPix2+FPix1_pos', 
            'BPix1+BPix2+FPix1_neg', 
            'BPix1+FPix1_pos+FPix2_pos', 
            'BPix1+FPix1_neg+FPix2_neg'
        ),
        MTOB = cms.PSet( ),
        TEC = cms.PSet( ),
        MTID = cms.PSet( ),
        FPix = cms.PSet( 
            HitProducer = cms.string( "hltSiPixelRecHits" ),
            hitErrorRZ = cms.double( 0.0036 ),
            useErrorsFromParam = cms.bool( True ),
            TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
            skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
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
            skipClusters = cms.InputTag( "hltIter1ClustersRefRemoval" ),
            hitErrorRPhi = cms.double( 0.0027 )
        ),
        TIB = cms.PSet( )
    )
    
    process.HLTIter1PSetTrajectoryFilterIT = cms.PSet( 
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        chargeSignificance = cms.double(-1.0),
        constantValueForLostHitsFractionFilter = cms.double(2.0),
        extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4),
        maxCCCLostHits = cms.int32(0), # offline (2),
        maxConsecLostHits = cms.int32(1),
        maxLostHits = cms.int32(1),  # offline (999),
        maxLostHitsFraction = cms.double(0.1),
        maxNumberOfHits = cms.int32(100),
        minGoodStripCharge = cms.PSet( refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
        minHitsMinPt = cms.int32(3),
        minNumberOfHitsForLoopers = cms.int32(13),
        minNumberOfHitsPerLoop = cms.int32(4),
        minPt = cms.double(0.2),
        minimumNumberOfHits = cms.int32(4), # 3 online
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
        maxCCCLostHits = cms.int32(0), # offline (2),
        maxConsecLostHits = cms.int32(1),
        maxLostHits = cms.int32(1),  # offline (999),
        maxLostHitsFraction = cms.double(0.1),
        maxNumberOfHits = cms.int32(100),
        minGoodStripCharge = cms.PSet( refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
        minHitsMinPt = cms.int32(3),
        minNumberOfHitsForLoopers = cms.int32(13),
        minNumberOfHitsPerLoop = cms.int32(4),
        minPt = cms.double(0.2),
        minimumNumberOfHits = cms.int32(4), # 3 online
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
        useSameTrajFilter = cms.bool(False) # new ! other iteration should have it set to True
    )

    process.HLTIterativeTrackingIteration1 = cms.Sequence( process.hltIter1ClustersRefRemoval + process.hltIter1MaskedMeasurementTrackerEvent + process.hltIter1PixelLayerTriplets + process.hltIter1PFlowPixelSeeds + process.hltIter1PFlowCkfTrackCandidates + process.hltIter1PFlowCtfWithMaterialTracks + process.hltIter1PFlowTrackCutClassifierPrompt + process.hltIter1PFlowTrackCutClassifierDetached + process.hltIter1PFlowTrackCutClassifierMerged + process.hltIter1PFlowTrackSelectionHighPurity )

    process.hltIter2PixelLayerTriplets = cms.EDProducer( "SeedingLayersEDProducer",
        layerList = cms.vstring( 
            'BPix1+BPix2+BPix3',
            'BPix1+BPix2+FPix1_pos',
            'BPix1+BPix2+FPix1_neg',
            'BPix1+FPix1_pos+FPix2_pos',
            'BPix1+FPix1_neg+FPix2_neg' 
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

    process.hltIter2PFlowPixelSeeds.OrderedHitsFactoryPSet = cms.PSet( 
        maxElement = cms.uint32( 0 ),
        ComponentName = cms.string( "StandardHitTripletGenerator" ),
        GeneratorPSet = cms.PSet( 
            useBending = cms.bool( True ),
            useFixedPreFiltering = cms.bool( False ),
            maxElement = cms.uint32( 100000 ),
            phiPreFiltering = cms.double( 0.3 ),
            extraHitRPhitolerance = cms.double( 0.032 ),
            useMultScattering = cms.bool( True ),
            ComponentName = cms.string( "PixelTripletHLTGenerator" ),
            extraHitRZtolerance = cms.double( 0.037 ),
            SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
        ),
        SeedingLayers = cms.InputTag( "hltIter2PixelLayerTriplets" )
    )

    process.hltIter2PFlowPixelSeeds.SeedCreatorPSet = cms.PSet(  refToPSet_ = cms.string( "HLTSeedFromConsecutiveHitsTripletOnlyCreator" ) )
    
    process.HLTIterativeTrackingIteration2 = cms.Sequence( process.hltIter2ClustersRefRemoval + process.hltIter2MaskedMeasurementTrackerEvent + process.hltIter2PixelLayerTriplets + process.hltIter2PFlowPixelSeeds + process.hltIter2PFlowCkfTrackCandidates + process.hltIter2PFlowCtfWithMaterialTracks + process.hltIter2PFlowTrackCutClassifier + process.hltIter2PFlowTrackSelectionHighPurity )

    for seqName in process.sequences:
        seq = getattr(process,seqName)
        from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor
        l = list()
        v = ModuleNodeVisitor(l)
        seq.visit(v)
        if process.hltPixelTracks in l and not process.hltPixelLayerQuadruplets in l:
            seq.remove(process.hltPixelLayerTriplets)
            index = seq.index(process.hltPixelTracks)
            seq.insert(index,process.hltPixelLayerQuadruplets)

    return process

# attach `customizeHLTForPFTrackingPhaseI2017` to the `phase1Pixel` era
def modifyHLTForPFTrackingPhaseI2017(process):
    phase1Pixel.toModify(process, customizeHLTForPFTrackingPhaseI2017)
