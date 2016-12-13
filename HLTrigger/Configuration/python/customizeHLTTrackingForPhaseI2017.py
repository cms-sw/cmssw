import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from HLTrigger.Configuration.common import *

def modifyHLTforPhaseIPixelGeom(fragment):
    """Modify the HLT configuration for the Phase I pixel geometry"""

    # modify all ClusterShapeHitFilterESProducer ESProducers
    for esproducer in esproducers_by_type(fragment, "ClusterShapeHitFilterESProducer"):
        trackingPhase1.toModify(esproducer, PixelShapeFile = 'RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1TkNewFPix.par')

    # modify all SiPixelRawToDigi EDProducers
    for producer in producers_by_type(fragment, "SiPixelRawToDigi"):
        trackingPhase1.toModify(producer, UsePhase1 = cms.bool( True ))


def modifyHLTforPhaseIPFTracking(fragment):
    """Modify the HLT configuration to run the Phase I tracking in the particle flow sequence"""

    # hltPixelLayerTriplets
    if hasattr(fragment, "hltPixelLayerTriplets"):
        trackingPhase1.toModify( fragment.hltPixelLayerTriplets,
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
            )
        )

    # hltPixelLayerQuadruplets
    trackingPhase1.toModify( fragment, lambda fragment:
        setattr(fragment, "hltPixelLayerQuadruplets", cms.EDProducer("SeedingLayersEDProducer",
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
        ) )
    )

    # hltPixelTracks
    if hasattr(fragment, "hltPixelTracks"):
        from RecoPixelVertexing.PixelTriplets.CAHitQuadrupletGenerator_cfi import CAHitQuadrupletGenerator as _CAHitQuadrupletGenerator
        trackingPhase1.toModify( fragment.hltPixelTracks, OrderedHitsFactoryPSet = _CAHitQuadrupletGenerator.clone(
            ComponentName = cms.string("CAHitQuadrupletGenerator"),
            extraHitRPhitolerance = cms.double(0.032),
            maxChi2 = dict(
                pt1     =   0.7,
                pt2     =   2,
                value1  = 200,
                value2  =  50,
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
        ) )
        trackingPhase1.toModify( fragment.hltPixelTracks.RegionFactoryPSet, RegionPSet = cms.PSet(
            precise = cms.bool( True ),
            beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
            originRadius = cms.double(0.02),
            ptMin = cms.double(0.9),
            nSigmaZ = cms.double(4.0),
        ) )

        # HLTDoRecoPixelTracksSequence
        trackingPhase1.toModify( fragment, lambda fragment:
            setattr(fragment, "HLTDoRecoPixelTracksSequence", cms.Sequence(
                fragment.hltPixelLayerQuadruplets +
                fragment.hltPixelTracks
            ) )
        )

    # HLTIter0PSetTrajectoryFilterIT
    if hasattr(fragment, "HLTIter0PSetTrajectoryFilterIT"):
        trackingPhase1.toModify( fragment.HLTIter0PSetTrajectoryFilterIT,
            minimumNumberOfHits = cms.int32( 4 ),
            minHitsMinPt        = cms.int32( 4 )
        )

    # hltIter0PFlowTrackCutClassifier
    if hasattr(fragment, "hltIter0PFlowTrackCutClassifier"):
        trackingPhase1.toModify( fragment.hltIter0PFlowTrackCutClassifier.mva,
            minLayers    = cms.vint32( 3, 3, 4 ),
            min3DLayers  = cms.vint32( 0, 3, 4 ),
            minPixelHits = cms.vint32( 0, 3, 4 )
        )

    # hltIter1PixelLayerTriplets
    trackingPhase1.toModify( fragment, lambda fragment:
        setattr(fragment, "hltIter1PixelLayerTriplets", cms.EDProducer( "SeedingLayersEDProducer",
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
        ) )
    )

    # HLTIter1PSetTrajectoryFilterIT
    trackingPhase1.toModify( fragment, lambda fragment:
        setattr(fragment, "HLTIter1PSetTrajectoryFilterIT", cms.PSet(
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
        ) )
    )

    # HLTIter1PSetTrajectoryFilterInOutIT
    trackingPhase1.toModify( fragment, lambda fragment:
        setattr(fragment, "HLTIter1PSetTrajectoryFilterInOutIT", cms.PSet(
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
        ) )
    )

    # HLTIter1PSetTrajectoryBuilderIT
    trackingPhase1.toModify( fragment, lambda fragment:
        setattr(fragment, "HLTIter1PSetTrajectoryBuilderIT", cms.PSet(
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
        ) )
    )

    # HLTIterativeTrackingIteration1
    trackingPhase1.toModify( fragment, lambda fragment:
        setattr(fragment, "HLTIterativeTrackingIteration1", cms.Sequence(
            fragment.hltIter1ClustersRefRemoval +
            fragment.hltIter1MaskedMeasurementTrackerEvent +
            fragment.hltIter1PixelLayerTriplets +
            fragment.hltIter1PFlowPixelSeeds +
            fragment.hltIter1PFlowCkfTrackCandidates +
            fragment.hltIter1PFlowCtfWithMaterialTracks +
            fragment.hltIter1PFlowTrackCutClassifierPrompt +
            fragment.hltIter1PFlowTrackCutClassifierDetached +
            fragment.hltIter1PFlowTrackCutClassifierMerged +
            fragment.hltIter1PFlowTrackSelectionHighPurity
        ) )
    )

    # hltIter2PixelLayerTriplets
    trackingPhase1.toModify( fragment, lambda fragment:
        setattr(fragment, "hltIter2PixelLayerTriplets", cms.EDProducer( "SeedingLayersEDProducer",
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
        ) )
    )

    # hltIter2PFlowPixelSeeds
    if hasattr(fragment, "hltIter2PFlowPixelSeeds"):
        trackingPhase1.toModify( fragment.hltIter2PFlowPixelSeeds,
            OrderedHitsFactoryPSet = cms.PSet(
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
            ),
            SeedCreatorPSet = cms.PSet(
                refToPSet_ = cms.string( "HLTSeedFromConsecutiveHitsTripletOnlyCreator" )
            )
        )

    # HLTIterativeTrackingIteration2
    trackingPhase1.toModify( fragment, lambda fragment:
        setattr(fragment, "HLTIterativeTrackingIteration2", cms.Sequence(
            fragment.hltIter2ClustersRefRemoval +
            fragment.hltIter2MaskedMeasurementTrackerEvent +
            fragment.hltIter2PixelLayerTriplets +
            fragment.hltIter2PFlowPixelSeeds +
            fragment.hltIter2PFlowCkfTrackCandidates +
            fragment.hltIter2PFlowCtfWithMaterialTracks +
            fragment.hltIter2PFlowTrackCutClassifier +
            fragment.hltIter2PFlowTrackSelectionHighPurity
        ) )
    )

    def add_hltPixelLayerQuadruplets_to_sequences(fragment):
        from FWCore.ParameterSet.SequenceTypes import ModuleNodeVisitor
        for seq_name in fragment.sequences:
            seq = getattr(fragment, seq_name)

            # find the list of modules in the sequence
            modules = list()
            seq.visit( ModuleNodeVisitor(modules) )

            if fragment.hltPixelTracks in modules and not fragment.hltPixelLayerQuadruplets in modules:
                mod = seq.copy()
                mod.remove(fragment.hltPixelLayerTriplets)
                index = mod.index(fragment.hltPixelTracks)
                mod.insert(index, fragment.hltPixelLayerQuadruplets)
                trackingPhase1.toReplaceWith(seq, mod)

    trackingPhase1.toModify( fragment, lambda fragment: add_hltPixelLayerQuadruplets_to_sequences(fragment) )
