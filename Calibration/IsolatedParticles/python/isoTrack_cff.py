import FWCore.ParameterSet.Config as cms

hltPreIsoTrackHE = cms.EDFilter("HLTPrescaler",
                                L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
                                offset = cms.uint32( 0 )
                                )
hltHITPixelTracksHBFitter = cms.EDProducer("PixelFitterByConformalMappingAndLineProducer",
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    useFixImpactParameter = cms.bool(True),
    fixImpactParameter = cms.double( 0.0 )
)
hltHITPixelTracksHBFilter = cms.EDProducer('PixelTrackFilterByKinematicsProducer',
    chi2 = cms.double( 1000.0 ),
    nSigmaTipMaxTolerance = cms.double( 0.0 ),
    nSigmaInvPtTolerance = cms.double( 0.0 ),
    ptMin = cms.double( 0.7 ),
    tipMax = cms.double( 1.0 )
)
hltHITPixelTracksCleaner = cms.ESProducer("PixelTrackCleanerBySharedHitsESProducer",
    ComponentName = cms.string("hltHITPixelTracksCleaner"),
    useQuadrupletAlgo = cms.bool(False)
)
hltHITPixelTracksHB = cms.EDProducer("PixelTrackProducer",
                                     Filter = cms.InputTag("hltHITPixelTracksHBFilter"),
                                     passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
                                     Fitter = cms.InputTag("hltHITPixelTracksHBFitter"),
                                     RegionFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "GlobalRegionProducerFromBeamSpot" ),
        RegionPSet = cms.PSet( 
            precise = cms.bool( True ),
            originRadius = cms.double( 0.0015 ),
            nSigmaZ = cms.double( 3.0 ),
            ptMin = cms.double( 0.7 ),
            beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
            )
        ),
                                     Cleaner = cms.string("hltHITPixelTracksCleaner"),
                                     OrderedHitsFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "StandardHitTripletGenerator" ),
        SeedingLayers = cms.string( "hltESPPixelLayerTripletsHITHB" ),
        GeneratorPSet = cms.PSet( 
            useBending = cms.bool( True ),
            useFixedPreFiltering = cms.bool( False ),
            maxElement = cms.uint32( 100000 ),
            phiPreFiltering = cms.double( 0.3 ),
            extraHitRPhitolerance = cms.double( 0.06 ),
            useMultScattering = cms.bool( True ),
            ComponentName = cms.string( "PixelTripletHLTGenerator" ),
            extraHitRZtolerance = cms.double( 0.06 ),
            SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
            )
        )
                                     )

hltHITPixelTracksHEFitter = cms.EDProducer("PixelFitterByConformalMappingAndLineProducer",
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    useFixImpactParameter = cms.bool(True),
    fixImpactParameter = cms.double( 0.0 )
)
hltHITPixelTracksHEFilter = cms.EDProducer('PixelTrackFilterByKinematicsProducer',
    chi2 = cms.double( 1000.0 ),
    nSigmaTipMaxTolerance = cms.double( 0.0 ),
    nSigmaInvPtTolerance = cms.double( 0.0 ),
    ptMin = cms.double( 0.35 ),
    tipMax = cms.double( 1.0 )
)
hltHITPixelTracksHE = cms.EDProducer("PixelTrackProducer",
                                     Filter = cms.InputTag("hltHITPixelTracksHEFilter"),
                                     passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
                                     Fitter = cms.InputTag("hltHITPixelTracksHEFitter"),
                                     RegionFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "GlobalRegionProducerFromBeamSpot" ),
        RegionPSet = cms.PSet( 
            precise = cms.bool( True ),
            originRadius = cms.double( 0.0015 ),
            nSigmaZ = cms.double( 3.0 ),
            beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
            ptMin = cms.double( 0.35 )
            )
        ),
                                     CleanerPSet = cms.PSet(
                                         ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ),
                                         useQuadrupletAlgo = cms.bool(False)
                                     ),
                                     OrderedHitsFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "StandardHitTripletGenerator" ),
        GeneratorPSet = cms.PSet( 
            useBending = cms.bool( True ),
            useFixedPreFiltering = cms.bool( False ),
            maxElement = cms.uint32( 100000 ),
            phiPreFiltering = cms.double( 0.3 ),
            extraHitRPhitolerance = cms.double( 0.06 ),
            useMultScattering = cms.bool( True ),
            ComponentName = cms.string( "PixelTripletHLTGenerator" ),
            extraHitRZtolerance = cms.double( 0.06 ),
            SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
            ),
        SeedingLayers = cms.string( "hltESPPixelLayerTripletsHITHE" )
        )
                                     )

hltHITPixelVerticesHE = cms.EDProducer("PixelVertexProducer",
                                       WtAverage = cms.bool( True ),
                                       Method2 = cms.bool( True ),
                                       beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
                                       Verbosity = cms.int32( 0 ),
                                       UseError = cms.bool( True ),
                                       TrackCollection = cms.InputTag( "hltHITPixelTracksHE" ),
                                       PtMin = cms.double( 1.0 ),
                                       NTrkMin = cms.int32( 2 ),
                                       ZOffset = cms.double( 5.0 ),
                                       Finder = cms.string( "DivisiveVertexFinder" ),
                                       ZSeparation = cms.double( 0.05 )
                                       )
hltIsolPixelTrackProdHE = cms.EDProducer("IsolatedPixelTrackCandidateProducer",
                                         minPTrack = cms.double( 5.0 ),
                                         L1eTauJetsSource = cms.InputTag( 'hltL1extraParticles','Tau' ),
                                         MaxVtxDXYSeed = cms.double( 101.0 ),
                                         tauUnbiasCone = cms.double( 1.2 ),
                                         VertexLabel = cms.InputTag( "hltHITPixelVerticesHE" ),
                                         L1GTSeedLabel = cms.InputTag( "hltL1sV0SingleJet60" ),
                                         EBEtaBoundary = cms.double( 1.479 ),
                                         maxPTrackForIsolation = cms.double( 3.0 ),
                                         MagFieldRecordName = cms.string( "VolumeBasedMagneticField" ),
                                         PixelIsolationConeSizeAtEC = cms.double( 40.0 ),
                                         PixelTracksSources = cms.VInputTag( 'hltHITPixelTracksHB','hltHITPixelTracksHE' ),
                                         MaxVtxDXYIsol = cms.double( 101.0 ),
                                         tauAssociationCone = cms.double( 0.0 ),
                                         ExtrapolationConeSize = cms.double( 1.0 )
                                         )

hltIsolPixelTrackL2FilterHE = cms.EDFilter("HLTPixelIsolTrackFilter",
                                           saveTags = cms.bool( True ),
                                           MaxPtNearby = cms.double( 2.0 ),
                                           MinEtaTrack = cms.double( 1.1 ),
                                           MinDeltaPtL1Jet = cms.double( -40000.0 ),
                                           MinPtTrack = cms.double( 3.5 ),
                                           DropMultiL2Event = cms.bool( False ),
                                           L1GTSeedLabel = cms.InputTag( "hltL1sV0SingleJet60" ),
                                           MinEnergyTrack = cms.double( 12.0 ),
                                           NMaxTrackCandidates = cms.int32( 5 ),
                                           MaxEtaTrack = cms.double( 2.2 ),
                                           candTag = cms.InputTag( "hltIsolPixelTrackProdHE" ),
                                           filterTrackEnergy = cms.bool( True )
                                           )

hltHITPixelTripletSeedGeneratorHE = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
                                                   RegionFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "HITRegionalPixelSeedGenerator" ),
        RegionPSet = cms.PSet( 
            useIsoTracks = cms.bool( True ),
            trackSrc = cms.InputTag( "hltHITPixelTracksHE" ),
            l1tjetSrc = cms.InputTag( 'hltL1extraParticles','Tau' ),
            isoTrackSrc = cms.InputTag( "hltIsolPixelTrackL2FilterHE" ),
            precise = cms.bool( True ),
            deltaEtaL1JetRegion = cms.double( 0.3 ),
            useTracks = cms.bool( False ),
            originRadius = cms.double( 0.6 ),
            originHalfLength = cms.double( 15.0 ),
            useL1Jets = cms.bool( False ),
            deltaPhiTrackRegion = cms.double( 0.05 ),
            deltaPhiL1JetRegion = cms.double( 0.3 ),
            vertexSrc = cms.string( "hltHITPixelVerticesHE" ),
            fixedReg = cms.bool( False ),
            etaCenter = cms.double( 0.0 ),
            phiCenter = cms.double( 0.0 ),
            originZPos = cms.double( 0.0 ),
            deltaEtaTrackRegion = cms.double( 0.05 ),
            ptMin = cms.double( 0.5 )
            )
        ),
                                                   SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
                                                   ClusterCheckPSet = cms.PSet( 
        PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
        MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
        doClusterCheck = cms.bool( False ),
        ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
        MaxNumberOfPixelClusters = cms.uint32( 10000 )
        ),
                                                   OrderedHitsFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "StandardHitTripletGenerator" ),
        GeneratorPSet = cms.PSet( 
            useBending = cms.bool( True ),
            useFixedPreFiltering = cms.bool( False ),
            maxElement = cms.uint32( 100000 ),
            ComponentName = cms.string( "PixelTripletHLTGenerator" ),
            extraHitRPhitolerance = cms.double( 0.06 ),
            useMultScattering = cms.bool( True ),
            phiPreFiltering = cms.double( 0.3 ),
            extraHitRZtolerance = cms.double( 0.06 ),
            SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
            ),
        SeedingLayers = cms.string( "hltESPPixelLayerTriplets" )
        ),
                                                   SeedCreatorPSet = cms.PSet( 
        ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
        propagator = cms.string( "PropagatorWithMaterial" )
        ),
                                                   TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
                                                   )

hltHITCkfTrackCandidatesHE = cms.EDProducer("CkfTrackCandidateMaker",
                                            src = cms.InputTag( "hltHITPixelTripletSeedGeneratorHE" ),
                                            maxSeedsBeforeCleaning = cms.uint32( 1000 ),
                                            TransientInitialStateEstimatorParameters = cms.PSet( 
        propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
        numberMeasurementsForFit = cms.int32( 4 ),
        propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
        ),
                                            TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
                                            cleanTrajectoryAfterInOut = cms.bool( False ),
                                            useHitsSplitting = cms.bool( False ),
                                            RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
                                            doSeedingRegionRebuilding = cms.bool( False ),
                                            maxNSeeds = cms.uint32( 100000 ),
                                            NavigationSchool = cms.string( "SimpleNavigationSchool" ),
                                            TrajectoryBuilder = cms.string( "hltESPCkfTrajectoryBuilder" )
                                            )
hltHITCtfWithMaterialTracksHE = cms.EDProducer("TrackProducer",
                                               src = cms.InputTag( "hltHITCkfTrackCandidatesHE" ),
                                               clusterRemovalInfo = cms.InputTag( "" ),
                                               beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
                                               Fitter = cms.string( "hltESPKFFittingSmoother" ),
                                               useHitsSplitting = cms.bool( False ),
                                               MeasurementTracker = cms.string( "" ),
                                               alias = cms.untracked.string( "hltHITCtfWithMaterialTracksHE8E29" ),
                                               NavigationSchool = cms.string( "" ),
                                               TrajectoryInEvent = cms.bool( False ),
                                               TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
                                               AlgorithmName = cms.string( "undefAlgorithm" ),
                                               Propagator = cms.string( "PropagatorWithMaterial" )
                                               )

hltHITIPTCorrectorHE = cms.EDProducer("IPTCorrector",
                                      corTracksLabel = cms.InputTag( "hltHITCtfWithMaterialTracksHE" ),
                                      filterLabel = cms.InputTag( "hltIsolPixelTrackL2FilterHE" ),
                                      associationCone = cms.double( 0.2 )
                                      )

hltIsolPixelTrackL3FilterHE = cms.EDFilter("HLTPixelIsolTrackFilter",
                                           saveTags = cms.bool( True ),
                                           MaxPtNearby = cms.double( 2.0 ),
                                           MinEtaTrack = cms.double( 1.1 ),
                                           MinDeltaPtL1Jet = cms.double( 4.0 ),
                                           MinPtTrack = cms.double( 20.0 ),
                                           DropMultiL2Event = cms.bool( False ),
                                           L1GTSeedLabel = cms.InputTag( "hltL1sV0SingleJet60" ),
                                           MinEnergyTrack = cms.double( 38.0 ),
                                           NMaxTrackCandidates = cms.int32( 999 ),
                                           MaxEtaTrack = cms.double( 2.2 ),
                                           candTag = cms.InputTag( "hltHITIPTCorrectorHE" ),
                                           filterTrackEnergy = cms.bool( True )
                                           )

hltPreIsoTrackHB = cms.EDFilter("HLTPrescaler",
                                L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
                                offset = cms.uint32( 0 )
                                )

hltHITPixelVerticesHB = cms.EDProducer("PixelVertexProducer",
                                       WtAverage = cms.bool( True ),
                                       Method2 = cms.bool( True ),
                                       beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
                                       Verbosity = cms.int32( 0 ),
                                       UseError = cms.bool( True ),
                                       TrackCollection = cms.InputTag( "hltHITPixelTracksHB" ),
                                       PtMin = cms.double( 1.0 ),
                                       NTrkMin = cms.int32( 2 ),
                                       ZOffset = cms.double( 5.0 ),
                                       Finder = cms.string( "DivisiveVertexFinder" ),
                                       ZSeparation = cms.double( 0.05 )
                                       )

hltIsolPixelTrackProdHB = cms.EDProducer("IsolatedPixelTrackCandidateProducer",
                                         minPTrack = cms.double( 5.0 ),
                                         L1eTauJetsSource = cms.InputTag( 'hltL1extraParticles','Tau' ),
                                         MaxVtxDXYSeed = cms.double( 101.0 ),
                                         tauUnbiasCone = cms.double( 1.2 ),
                                         VertexLabel = cms.InputTag( "hltHITPixelVerticesHB" ),
                                         L1GTSeedLabel = cms.InputTag( "hltL1sV0SingleJet60" ),
                                         EBEtaBoundary = cms.double( 1.479 ),
                                         maxPTrackForIsolation = cms.double( 3.0 ),
                                         MagFieldRecordName = cms.string( "VolumeBasedMagneticField" ),
                                         PixelIsolationConeSizeAtEC = cms.double( 40.0 ),
                                         PixelTracksSources = cms.VInputTag( 'hltHITPixelTracksHB' ),
                                         MaxVtxDXYIsol = cms.double( 101.0 ),
                                         tauAssociationCone = cms.double( 0.0 ),
                                         ExtrapolationConeSize = cms.double( 1.0 )
                                         )

hltIsolPixelTrackL2FilterHB = cms.EDFilter("HLTPixelIsolTrackFilter",
                                           saveTags = cms.bool( True ),
                                           MaxPtNearby = cms.double( 2.0 ),
                                           MinEtaTrack = cms.double( 0.0 ),
                                           MinDeltaPtL1Jet = cms.double( -40000.0 ),
                                           MinPtTrack = cms.double( 3.5 ),
                                           DropMultiL2Event = cms.bool( False ),
                                           L1GTSeedLabel = cms.InputTag( "hltL1sV0SingleJet60" ),
                                           MinEnergyTrack = cms.double( 8.0 ),
                                           NMaxTrackCandidates = cms.int32( 10 ),
                                           MaxEtaTrack = cms.double( 1.15 ),
                                           candTag = cms.InputTag( "hltIsolPixelTrackProdHB" ),
                                           filterTrackEnergy = cms.bool( True )
                                           )

hltHITPixelTripletSeedGeneratorHB = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
                                                   RegionFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "HITRegionalPixelSeedGenerator" ),
        RegionPSet = cms.PSet( 
            useIsoTracks = cms.bool( True ),
            trackSrc = cms.InputTag( "hltHITPixelTracksHB" ),
            l1tjetSrc = cms.InputTag( 'hltL1extraParticles','Tau' ),
            isoTrackSrc = cms.InputTag( "hltIsolPixelTrackL2FilterHB" ),
            precise = cms.bool( True ),
            deltaEtaL1JetRegion = cms.double( 0.3 ),
            useTracks = cms.bool( False ),
            originRadius = cms.double( 0.6 ),
            originHalfLength = cms.double( 15.0 ),
            useL1Jets = cms.bool( False ),
            deltaPhiTrackRegion = cms.double( 0.05 ),
            deltaPhiL1JetRegion = cms.double( 0.3 ),
            vertexSrc = cms.string( "hltHITPixelVerticesHB" ),
            fixedReg = cms.bool( False ),
            etaCenter = cms.double( 0.0 ),
            phiCenter = cms.double( 0.0 ),
            originZPos = cms.double( 0.0 ),
            deltaEtaTrackRegion = cms.double( 0.05 ),
            ptMin = cms.double( 1.0 )
            )
        ),
                                                   SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
                                                   ClusterCheckPSet = cms.PSet( 
        PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
        MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
        doClusterCheck = cms.bool( False ),
        ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
        MaxNumberOfPixelClusters = cms.uint32( 10000 )
        ),
                                                   OrderedHitsFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "StandardHitTripletGenerator" ),
        GeneratorPSet = cms.PSet( 
            useBending = cms.bool( True ),
            useFixedPreFiltering = cms.bool( False ),
            maxElement = cms.uint32( 100000 ),
            ComponentName = cms.string( "PixelTripletHLTGenerator" ),
            extraHitRPhitolerance = cms.double( 0.06 ),
            useMultScattering = cms.bool( True ),
            phiPreFiltering = cms.double( 0.3 ),
            extraHitRZtolerance = cms.double( 0.06 ),
            SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
            ),
        SeedingLayers = cms.string( "hltESPPixelLayerTriplets" )
        ),
                                                   SeedCreatorPSet = cms.PSet( 
        ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
        propagator = cms.string( "PropagatorWithMaterial" )
        ),
                                                   TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
                                                   )

hltHITCkfTrackCandidatesHB = cms.EDProducer("CkfTrackCandidateMaker",
                                            src = cms.InputTag( "hltHITPixelTripletSeedGeneratorHB" ),
                                            maxSeedsBeforeCleaning = cms.uint32( 1000 ),
                                            TransientInitialStateEstimatorParameters = cms.PSet( 
        propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
        numberMeasurementsForFit = cms.int32( 4 ),
        propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
        ),
                                            TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
                                            cleanTrajectoryAfterInOut = cms.bool( False ),
                                            useHitsSplitting = cms.bool( False ),
                                            RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
                                            doSeedingRegionRebuilding = cms.bool( False ),
                                            maxNSeeds = cms.uint32( 100000 ),
                                            NavigationSchool = cms.string( "SimpleNavigationSchool" ),
                                            TrajectoryBuilder = cms.string( "hltESPCkfTrajectoryBuilder" )
                                            )

hltHITCtfWithMaterialTracksHB = cms.EDProducer("TrackProducer",
                                               src = cms.InputTag( "hltHITCkfTrackCandidatesHB" ),
                                               clusterRemovalInfo = cms.InputTag( "" ),
                                               beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
                                               Fitter = cms.string( "hltESPKFFittingSmoother" ),
                                               useHitsSplitting = cms.bool( False ),
                                               MeasurementTracker = cms.string( "" ),
                                               alias = cms.untracked.string( "hltHITCtfWithMaterialTracksHB8E29" ),
                                               NavigationSchool = cms.string( "" ),
                                               TrajectoryInEvent = cms.bool( False ),
                                               TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
                                               AlgorithmName = cms.string( "undefAlgorithm" ),
                                               Propagator = cms.string( "PropagatorWithMaterial" )
                                               )

hltHITIPTCorrectorHB = cms.EDProducer("IPTCorrector",
                                      corTracksLabel = cms.InputTag( "hltHITCtfWithMaterialTracksHB" ),
                                      filterLabel = cms.InputTag( "hltIsolPixelTrackL2FilterHB" ),
                                      associationCone = cms.double( 0.2 )
                                      )

hltIsolPixelTrackL3FilterHB = cms.EDFilter("HLTPixelIsolTrackFilter",
                                           saveTags = cms.bool( True ),
                                           MaxPtNearby = cms.double( 2.0 ),
                                           MinEtaTrack = cms.double( 0.0 ),
                                           MinDeltaPtL1Jet = cms.double( 4.0 ),
                                           MinPtTrack = cms.double( 20.0 ),
                                           DropMultiL2Event = cms.bool( False ),
                                           L1GTSeedLabel = cms.InputTag( "hltL1sV0SingleJet60" ),
                                           MinEnergyTrack = cms.double( 38.0 ),
                                           NMaxTrackCandidates = cms.int32( 999 ),
                                            MaxEtaTrack = cms.double( 1.15 ),
                                           candTag = cms.InputTag( "hltHITIPTCorrectorHB" ),
                                           filterTrackEnergy = cms.bool( True )
                                           )

hltEcalIsolPixelTrackL2FilterHB = cms.EDFilter("HLTEcalPixelIsolTrackFilter",
                                               MaxEnergyIn = cms.double(1.2),
                                               MaxEnergyOut = cms.double(1.2),
                                               candTag = cms.InputTag("isolEcalPixelTrackProdHB"),
                                               NMaxTrackCandidates=cms.int32(10),
                                               DropMultiL2Event = cms.bool(False),
                                               saveTags = cms.bool( False )
                                               )

hltEcalIsolPixelTrackL2FilterHE = cms.EDFilter("HLTEcalPixelIsolTrackFilter",
                                               MaxEnergyIn = cms.double(1.2),
                                               MaxEnergyOut = cms.double(1.2),
                                               candTag = cms.InputTag("isolEcalPixelTrackProdHE"),
                                               NMaxTrackCandidates=cms.int32(10),
                                               DropMultiL2Event = cms.bool(False),
                                               saveTags = cms.bool( False )
                                               )

HLT_IsoTrackHE_v15 = cms.Path( HLTBeginSequence + hltL1sV0SingleJet60 + hltPreIsoTrackHE + HLTDoLocalPixelSequence + hltHITPixelTracksHBFitter + hltHITPixelTracksHBFilter + hltHITPixelTracksHEFitter + hltHITPixelTracksHEFilter + hltHITPixelTracksHB + hltHITPixelTracksHE + hltHITPixelVerticesHE + hltIsolPixelTrackProdHE + hltIsolPixelTrackL2FilterHE + HLTDoLocalStripSequence + hltHITPixelTripletSeedGeneratorHE + hltHITCkfTrackCandidatesHE + hltHITCtfWithMaterialTracksHE + hltHITIPTCorrectorHE + hltIsolPixelTrackL3FilterHE + HLTEndSequence )

HLT_IsoTrackHB_v14 = cms.Path( HLTBeginSequence + hltL1sV0SingleJet60 + hltPreIsoTrackHB + HLTDoLocalPixelSequence + hltHITPixelTracksHBFitter + hltHITPixelTracksHBFilter + hltHITPixelTracksHB + hltHITPixelVerticesHB + hltIsolPixelTrackProdHB + hltIsolPixelTrackL2FilterHB + HLTDoLocalStripSequence + hltHITPixelTripletSeedGeneratorHB + hltHITCkfTrackCandidatesHB + hltHITCtfWithMaterialTracksHB + hltHITIPTCorrectorHB + hltIsolPixelTrackL3FilterHB + HLTEndSequence )

