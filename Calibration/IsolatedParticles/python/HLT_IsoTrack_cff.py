import FWCore.ParameterSet.Config as cms

hltPreIsoTrackHE = cms.EDFilter("HLTPrescaler",
                                L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
                                offset = cms.uint32( 0 )
                                )

hltIsolPixelTrackProdHE = cms.EDProducer("IsolatedPixelTrackCandidateProducer",
                                         minPTrack = cms.double( 5.0 ),
                                         L1eTauJetsSource = cms.InputTag( 'hltL1extraParticles','Tau' ),
                                         MaxVtxDXYSeed = cms.double( 101.0 ),
                                         tauUnbiasCone = cms.double( 1.2 ),
                                         VertexLabel = cms.InputTag( "hltTrimmedPixelVertices" ),
                                         L1GTSeedLabel = cms.InputTag( "hltL1sL1SingleJet68" ),
                                         EBEtaBoundary = cms.double( 1.479 ),
                                         maxPTrackForIsolation = cms.double( 3.0 ),
                                         MagFieldRecordName = cms.string( "VolumeBasedMagneticField" ),
                                         PixelIsolationConeSizeAtEC = cms.double( 40.0 ),
                                         PixelTracksSources = cms.VInputTag( 'hltPixelTracks' ),
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

hltIsolEcalPixelTrackProdHE = cms.EDProducer("IsolatedEcalPixelTrackCandidateProducer",
                                             filterLabel               = cms.InputTag("hltIsolPixelTrackL2FilterHE"),
                                             EBRecHitSource = cms.InputTag('hltEcalRecHit','EcalRecHitsEB'),
                                             EERecHitSource = cms.InputTag('hltEcalRecHit','EcalRecHitsEE'),
                                             ECHitEnergyThreshold      = cms.double(0.05),
                                             ECHitCountEnergyThreshold = cms.double(0.5),
                                             EcalConeSizeEta0          = cms.double(0.09),
                                             EcalConeSizeEta1          = cms.double(0.14)
                                        )

hltEcalIsolPixelTrackL2FilterHE = cms.EDFilter("HLTEcalPixelIsolTrackFilter",
                                               MaxEnergyIn = cms.double(1.2),
                                               MaxEnergyOut = cms.double(1.2),
                                               candTag = cms.InputTag("hltIsolEcalPixelTrackProdHE"),
                                               NMaxTrackCandidates=cms.int32(10),
                                               DropMultiL2Event = cms.bool(False),
                                               saveTags = cms.bool( False )
                                               )

hltHcalITIPTCorrectorHE = cms.EDProducer("IPTCorrector",
                                         corTracksLabel = cms.InputTag( "hltIter0PFlowCtfWithMaterialTracks" ),
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
                                           MinEnergyTrack = cms.double( 18.0 ),
                                           NMaxTrackCandidates = cms.int32( 999 ),
                                           MaxEtaTrack = cms.double( 2.2 ),
                                           candTag = cms.InputTag( "hltHcalITIPTCorrectorHE" ),
                                           filterTrackEnergy = cms.bool( True )
                                           )

hltPreIsoTrackHB = cms.EDFilter("HLTPrescaler",
                                L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
                                offset = cms.uint32( 0 )
                                )

hltIsolPixelTrackProdHB = cms.EDProducer("IsolatedPixelTrackCandidateProducer",
                                         minPTrack = cms.double( 5.0 ),
                                         L1eTauJetsSource = cms.InputTag( 'hltL1extraParticles','Tau' ),
                                         MaxVtxDXYSeed = cms.double( 101.0 ),
                                         tauUnbiasCone = cms.double( 1.2 ),
                                         VertexLabel = cms.InputTag( "hltTrimmedPixelVertices" ),
                                         L1GTSeedLabel = cms.InputTag( "hltL1sL1SingleJet68" ),
                                         EBEtaBoundary = cms.double( 1.479 ),
                                         maxPTrackForIsolation = cms.double( 3.0 ),
                                         MagFieldRecordName = cms.string( "VolumeBasedMagneticField" ),
                                         PixelIsolationConeSizeAtEC = cms.double( 40.0 ),
                                         PixelTracksSources = cms.VInputTag( 'hltPixelTracks' ),
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
                                           MinEnergyTrack = cms.double( 12.0 ),
                                           NMaxTrackCandidates = cms.int32( 10 ),
                                           MaxEtaTrack = cms.double( 1.15 ),
                                           candTag = cms.InputTag( "hltIsolPixelTrackProdHB" ),
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

hltEcalIsolPixelTrackL2FilterHB = cms.EDFilter("HLTEcalPixelIsolTrackFilter",
                                               MaxEnergyIn = cms.double(1.2),
                                               MaxEnergyOut = cms.double(1.2),
                                               candTag = cms.InputTag("hltIsolEcalPixelTrackProdHB"),
                                               NMaxTrackCandidates=cms.int32(10),
                                               DropMultiL2Event = cms.bool(False),
                                               saveTags = cms.bool( False )
                                               )

hltHcalITIPTCorrectorHB = cms.EDProducer("IPTCorrector",
                                         corTracksLabel = cms.InputTag( "hltIter0PFlowCtfWithMaterialTracks" ),
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
                                           MinEnergyTrack = cms.double( 18.0 ),
                                           NMaxTrackCandidates = cms.int32( 999 ),
                                           MaxEtaTrack = cms.double( 1.15 ),
                                           candTag = cms.InputTag( "hltHcalITIPTCorrectorHB" ),
                                           filterTrackEnergy = cms.bool( True )
                                           )
