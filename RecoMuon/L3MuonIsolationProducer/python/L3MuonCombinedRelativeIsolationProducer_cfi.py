import FWCore.ParameterSet.Config as cms


hltL3MuonIsolations = cms.EDProducer( "L3MuonCombinedRelativeIsolationProducer",
                                                     inputMuonCollection = cms.InputTag( "hltL3Muons" ),
                                                     OutputMuIsoDeposits = cms.bool( True ),
                                                     TrackPt_Min = cms.double( -1.0 ),
                                                     printDebug = cms.bool(False),
                                                     CutsPSet = cms.PSet(
                                                        ConeSizes = cms.vdouble( 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24 ),
                                                        ComponentName = cms.string( "SimpleCuts" ),
                                                        Thresholds = cms.vdouble( 1000000.1, 1000000.1, 100000000.1, 1000000.1, 1000000.2, 1000000.1, 1000000.2, 10000000.1, 1000000.2, 1000000.0, 10000000.1, 1000000.0, 10000000.0, 10000000.1, 1000000.0, 1000000.0, 1000000.1, 1000000.9, 1000000.1, 1000000.9, 1000000.1, 1000000.0, 1000000.0, 1000000.9, 1000000.8, 1000000.1 ),
                                                        maxNTracks = cms.int32( -1 ),
                                                        EtaBounds = cms.vdouble( 0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 1.785, 1.88, 1.9865, 2.1075, 2.247, 2.411 ),
                                                        applyCutsORmaxNTracks = cms.bool( False )
                                                     ),
                                                     TrkExtractorPSet = cms.PSet(
              Chi2Prob_Min = cms.double( -1.0 ),
                      Diff_z = cms.double( 0.2 ),
                      inputTrackCollection = cms.InputTag( "hltPixelTracks" ),
                      ReferenceRadius = cms.double( 6.0 ),
                      BeamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
                      ComponentName = cms.string( "PixelTrackExtractor" ),
                      DR_Max = cms.double( 0.24 ),
                      Diff_r = cms.double( 0.1 ),
                      VetoLeadingTrack = cms.bool( True ),
                      DR_VetoPt = cms.double( 0.025 ),
                      DR_Veto = cms.double( 0.01 ),
                      NHits_Min = cms.uint32( 0 ),
                      Chi2Ndof_Max = cms.double( 1.0E64 ),
                      Pt_Min = cms.double( -1.0 ),
                      DepositLabel = cms.untracked.string( "PXLS" ),
                      BeamlineOption = cms.string( "BeamSpotFromEvent" ),
                      PropagateTracksToRadius = cms.bool( True ),
                      PtVeto_Min = cms.double( 2.0 )
                      ),
                                                    CaloExtractorPSet = cms.PSet(
            DR_Veto_H = cms.double( 0.1 ),
                  Vertex_Constraint_Z = cms.bool( False ),
                  Threshold_H = cms.double( 0.5 ),
                  ComponentName = cms.string( "CaloExtractor" ),
                  Threshold_E = cms.double( 0.2 ),
                  DR_Max = cms.double( 0.24 ),
                  DR_Veto_E = cms.double( 0.07 ),
                  Weight_E = cms.double( 1.5 ),
                  Vertex_Constraint_XY = cms.bool( False ),
                  DepositLabel = cms.untracked.string( "EcalPlusHcal" ),
                  CaloTowerCollectionLabel = cms.InputTag( "hltTowerMakerForMuons" ),
                  Weight_H = cms.double( 1.0 )
                )
               )
