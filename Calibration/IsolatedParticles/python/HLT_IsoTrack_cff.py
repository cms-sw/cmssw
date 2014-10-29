import FWCore.ParameterSet.Config as cms

hltL1sL1SingleJet68 = cms.EDFilter("HLTLevel1GTSeed",
                                   L1SeedsLogicalExpression = cms.string( "L1_SingleJet68" ),
                                   saveTags = cms.bool( True ),
                                   L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
                                   L1UseL1TriggerObjectMaps = cms.bool( True ),
                                   L1UseAliasesForSeeding = cms.bool( True ),
                                   L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
                                   L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
                                   L1NrBxInEvent = cms.int32( 3 ),
                                   L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
                                   L1TechTriggerSeeding = cms.bool( False )
                                   )

hltPreIsoTrackHE = cms.EDFilter("HLTPrescaler",
                                L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
                                offset = cms.uint32( 0 )
                                )

hltPixelLayerTripletsHITHB = cms.EDProducer("SeedingLayersEDProducer",
                                            layerList = cms.vstring( 'BPix1+BPix2+BPix3' ),
                                            MTOB = cms.PSet(  ),
                                            TEC = cms.PSet(  ),
                                            MTID = cms.PSet(  ),
                                            FPix = cms.PSet( 
        useErrorsFromParam = cms.bool( True ),
        hitErrorRPhi = cms.double( 0.0051 ),
        TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
        HitProducer = cms.string( "hltSiPixelRecHits" ),
        hitErrorRZ = cms.double( 0.0036 )
        ),
                                            MTEC = cms.PSet(  ),
                                            MTIB = cms.PSet(  ),
                                            TID = cms.PSet(  ),
                                            TOB = cms.PSet(  ),
                                            BPix = cms.PSet( 
        useErrorsFromParam = cms.bool( True ),
        hitErrorRPhi = cms.double( 0.0027 ),
        TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
        HitProducer = cms.string( "hltSiPixelRecHits" ),
        hitErrorRZ = cms.double( 0.006 )
        ),
                                            TIB = cms.PSet(  )
                                            )

hltHITPixelTracksHB = cms.EDProducer("PixelTrackProducer",
                                     FilterPSet = cms.PSet( 
        chi2 = cms.double( 1000.0 ),
        nSigmaTipMaxTolerance = cms.double( 0.0 ),
        ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
        nSigmaInvPtTolerance = cms.double( 0.0 ),
        ptMin = cms.double( 0.7 ),
        tipMax = cms.double( 1.0 )
        ),
                                     useFilterWithES = cms.bool( False ),
                                     passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
                                     FitterPSet = cms.PSet( 
        ComponentName = cms.string( "PixelFitterByConformalMappingAndLine" ),
        TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
        fixImpactParameter = cms.double( 0.0 )
        ),
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
                                     CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
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
        SeedingLayers = cms.InputTag( "hltPixelLayerTripletsHITHB" )
        )
                                      )
hltPixelLayerTripletsHITHE = cms.EDProducer("SeedingLayersEDProducer",
                                            layerList = cms.vstring( 'BPix1+BPix2+FPix1_pos',
                                                                     'BPix1+BPix2+FPix1_neg',
                                                                     'BPix1+FPix1_pos+FPix2_pos',
                                                                     'BPix1+FPix1_neg+FPix2_neg' ),
                                            MTOB = cms.PSet(  ),
                                            TEC = cms.PSet(  ),
                                            MTID = cms.PSet(  ),
                                            FPix = cms.PSet( 
        useErrorsFromParam = cms.bool( True ),
        hitErrorRPhi = cms.double( 0.0051 ),
        TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
        HitProducer = cms.string( "hltSiPixelRecHits" ),
        hitErrorRZ = cms.double( 0.0036 )
        ),
                                            MTEC = cms.PSet(  ),
                                            MTIB = cms.PSet(  ),
                                            TID = cms.PSet(  ),
                                            TOB = cms.PSet(  ),
                                            BPix = cms.PSet( 
        useErrorsFromParam = cms.bool( True ),
        hitErrorRPhi = cms.double( 0.0027 ),
        TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
        HitProducer = cms.string( "hltSiPixelRecHits" ),
        hitErrorRZ = cms.double( 0.006 )
        ),
                                            TIB = cms.PSet(  )
                                            )

hltHITPixelTracksHE = cms.EDProducer("PixelTrackProducer",
                                     FilterPSet = cms.PSet( 
        chi2 = cms.double( 1000.0 ),
        nSigmaTipMaxTolerance = cms.double( 0.0 ),
        ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
        nSigmaInvPtTolerance = cms.double( 0.0 ),
        ptMin = cms.double( 0.35 ),
        tipMax = cms.double( 1.0 )
        ),
                                     useFilterWithES = cms.bool( False ),
                                     passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
                                     FitterPSet = cms.PSet( 
        ComponentName = cms.string( "PixelFitterByConformalMappingAndLine" ),
        TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
        fixImpactParameter = cms.double( 0.0 )
        ),
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
                                     CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
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
        SeedingLayers = cms.InputTag( "hltPixelLayerTripletsHITHE" )
        )
                                     )

hltHITPixelVerticesHE = cms.EDProducer("PixelVertexProducer",
                                       WtAverage = cms.bool( True ),
                                       Method2 = cms.bool( True ),
                                       beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
                                       PVcomparer = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPvClusterComparer" ) ),
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
                                         L1GTSeedLabel = cms.InputTag( "hltL1sL1SingleJet68" ),
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
                                           MaxPtNearby = cms.double( 2.0 ),
                                           saveTags = cms.bool( True ),
                                           MinEtaTrack = cms.double( 1.1 ),
                                           MinDeltaPtL1Jet = cms.double( -40000.0 ),
                                           MinPtTrack = cms.double( 3.5 ),
                                           DropMultiL2Event = cms.bool( False ),
                                           L1GTSeedLabel = cms.InputTag( "hltL1sL1SingleJet68" ),
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
            fixedReg = cms.bool( False ),
            etaCenter = cms.double( 0.0 ),
            phiCenter = cms.double( 0.0 ),
            originZPos = cms.double( 0.0 ),
            deltaEtaTrackRegion = cms.double( 0.05 ),
            ptMin = cms.double( 0.5 ),
            vertexSrc = cms.InputTag( "hltHITPixelVerticesHE" )
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
        SeedingLayers = cms.InputTag( "hltPixelLayerTriplets" )
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
                                            SimpleMagneticField = cms.string( "" ),
                                            TransientInitialStateEstimatorParameters = cms.PSet( 
        propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
        numberMeasurementsForFit = cms.int32( 4 ),
        propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
        ),
                                            TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
                                            MeasurementTrackerEvent = cms.InputTag( "hltSiStripClusters" ),
                                            cleanTrajectoryAfterInOut = cms.bool( False ),
                                            useHitsSplitting = cms.bool( False ),
                                            RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
                                            doSeedingRegionRebuilding = cms.bool( False ),
                                            maxNSeeds = cms.uint32( 100000 ),
                                            TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfTrajectoryBuilder" ) ),
                                            NavigationSchool = cms.string( "SimpleNavigationSchool" ),
                                            TrajectoryBuilder = cms.string( "hltESPCkfTrajectoryBuilder" )
                                            )

hltHITCtfWithMaterialTracksHE = cms.EDProducer("TrackProducer",
                                               src = cms.InputTag( "hltHITCkfTrackCandidatesHE" ),
                                               SimpleMagneticField = cms.string( "" ),
                                               clusterRemovalInfo = cms.InputTag( "" ),
                                               beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
                                               MeasurementTrackerEvent = cms.InputTag( "hltSiStripClusters" ),
                                               Fitter = cms.string( "hltESPKFFittingSmoother" ),
                                               useHitsSplitting = cms.bool( False ),
                                               MeasurementTracker = cms.string( "" ),
                                               AlgorithmName = cms.string( "undefAlgorithm" ),
                                               alias = cms.untracked.string( "hltHITCtfWithMaterialTracksHE8E29" ),
                                               NavigationSchool = cms.string( "" ),
                                               TrajectoryInEvent = cms.bool( False ),
                                               TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
                                               GeometricInnerState = cms.bool( True ),
                                               useSimpleMF = cms.bool( False ),
                                               Propagator = cms.string( "PropagatorWithMaterial" )
                                               )

hltHITIPTCorrectorHE = cms.EDProducer("IPTCorrector",
                                      corTracksLabel = cms.InputTag( "hltHITCtfWithMaterialTracksHE" ),
                                      filterLabel = cms.InputTag( "hltIsolPixelTrackL2FilterHE" ),
                                      associationCone = cms.double( 0.2 )
                                      )

hltIsolPixelTrackL3FilterHE = cms.EDFilter("HLTPixelIsolTrackFilter",
                                           MaxPtNearby = cms.double( 2.0 ),
                                           saveTags = cms.bool( True ),
                                           MinEtaTrack = cms.double( 1.1) ,
                                           MinDeltaPtL1Jet = cms.double( 4.0 ),
                                           MinPtTrack = cms.double( 20.0 ),
                                           DropMultiL2Event = cms.bool( False ),
                                           L1GTSeedLabel = cms.InputTag( "hltL1sL1SingleJet68" ),
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
                                       PVcomparer = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPvClusterComparer" ) ),
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
                                         L1GTSeedLabel = cms.InputTag( "hltL1sL1SingleJet68" ),
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
                                           MaxPtNearby = cms.double( 2.0 ),
                                           saveTags = cms.bool( True ),
                                           MinEtaTrack = cms.double( 0.0 ),
                                           MinDeltaPtL1Jet = cms.double( -40000.0 ),
                                           MinPtTrack = cms.double( 3.5 ),
                                           DropMultiL2Event = cms.bool( False ),
                                           L1GTSeedLabel = cms.InputTag( "hltL1sL1SingleJet68" ),
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
            fixedReg = cms.bool( False ),
            etaCenter = cms.double( 0.0 ),
            phiCenter = cms.double( 0.0 ),
            originZPos = cms.double( 0.0 ),
            deltaEtaTrackRegion = cms.double( 0.05 ),
            ptMin = cms.double( 1.0 ),
            vertexSrc = cms.InputTag( "hltHITPixelVerticesHB" )
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
        SeedingLayers = cms.InputTag( "hltPixelLayerTriplets" )
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
                                            SimpleMagneticField = cms.string( "" ),
                                            TransientInitialStateEstimatorParameters = cms.PSet( 
        propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
        numberMeasurementsForFit = cms.int32( 4 ),
        propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
        ),
                                            TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
                                            MeasurementTrackerEvent = cms.InputTag( "hltSiStripClusters" ),
                                            cleanTrajectoryAfterInOut = cms.bool( False ),
                                            useHitsSplitting = cms.bool( False ),
                                            RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
                                            doSeedingRegionRebuilding = cms.bool( False ),
                                            maxNSeeds = cms.uint32( 100000 ),
                                            TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfTrajectoryBuilder" ) ),
                                            NavigationSchool = cms.string( "SimpleNavigationSchool" ),
                                            TrajectoryBuilder = cms.string( "hltESPCkfTrajectoryBuilder" )
                                            )

hltHITCtfWithMaterialTracksHB = cms.EDProducer("TrackProducer",
                                               src = cms.InputTag( "hltHITCkfTrackCandidatesHB" ),
                                               SimpleMagneticField = cms.string( "" ),
                                               clusterRemovalInfo = cms.InputTag( "" ),
                                               beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
                                               MeasurementTrackerEvent = cms.InputTag( "hltSiStripClusters" ),
                                               Fitter = cms.string( "hltESPKFFittingSmoother" ),
                                               useHitsSplitting = cms.bool( False ),
                                               MeasurementTracker = cms.string( "" ),
                                               AlgorithmName = cms.string( "undefAlgorithm" ),
                                               alias = cms.untracked.string( "hltHITCtfWithMaterialTracksHB8E29" ),
                                               NavigationSchool = cms.string( "" ),
                                               TrajectoryInEvent = cms.bool( False ),
                                               TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
                                               GeometricInnerState = cms.bool( True ),
                                               useSimpleMF = cms.bool( False ),
                                               Propagator = cms.string( "PropagatorWithMaterial" )
                                               )

hltHITIPTCorrectorHB = cms.EDProducer("IPTCorrector",
                                      corTracksLabel = cms.InputTag( "hltHITCtfWithMaterialTracksHB" ),
                                      filterLabel = cms.InputTag( "hltIsolPixelTrackL2FilterHB" ),
                                      associationCone = cms.double( 0.2 )
                                      )

hltIsolPixelTrackL3FilterHB = cms.EDFilter("HLTPixelIsolTrackFilter",
                                           MaxPtNearby = cms.double( 2.0 ),
                                           saveTags = cms.bool( True ),
                                           MinEtaTrack = cms.double( 0.0 ),
                                           MinDeltaPtL1Jet = cms.double( 4.0 ),
                                           MinPtTrack = cms.double( 20.0 ),
                                           DropMultiL2Event = cms.bool( False ),
                                           L1GTSeedLabel = cms.InputTag( "hltL1sL1SingleJet68" ),
                                           MinEnergyTrack = cms.double( 38.0 ),
                                           NMaxTrackCandidates = cms.int32( 999 ),
                                           MaxEtaTrack = cms.double( 1.15 ),
                                           candTag = cms.InputTag( "hltHITIPTCorrectorHB" ),
                                           filterTrackEnergy = cms.bool( True )
                                           )

hltIsolEcalPixelTrackProdHB = cms.EDProducer("IsolatedEcalPixelTrackCandidateProducer",
                                             filterLabel               = cms.InputTag("hltIsolPixelTrackL2FilterHB"),
                                             EBRecHitSource = cms.InputTag('hltEcalRecHit','EcalRecHitsEB'),
                                             EERecHitSource = cms.InputTag('hltEcalRecHit','EcalRecHitsEE'),
                                             ECHitEnergyThreshold      = cms.double(0.05),
                                             ECHitCountEnergyThreshold = cms.double(0.5),
                                             EcalConeSizeEta0          = cms.double(0.09),
                                             EcalConeSizeEta1          = cms.double(0.14)
                                        )

hltIsolEcalPixelTrackProdHE = cms.EDProducer("IsolatedEcalPixelTrackCandidateProducer",
                                             filterLabel               = cms.InputTag("hltIsolPixelTrackL2FilterHE"),
                                             EBRecHitSource = cms.InputTag('hltEcalRecHit','EcalRecHitsEB'),
                                             EERecHitSource = cms.InputTag('hltEcalRecHit','EcalRecHitsEE'),
                                             ECHitEnergyThreshold      = cms.double(0.05),
                                             ECHitCountEnergyThreshold = cms.double(0.5),
                                             EcalConeSizeEta0          = cms.double(0.09),
                                             EcalConeSizeEta1          = cms.double(0.14)
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
