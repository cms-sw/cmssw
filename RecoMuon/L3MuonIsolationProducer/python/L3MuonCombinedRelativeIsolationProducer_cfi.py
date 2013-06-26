import FWCore.ParameterSet.Config as cms


hltL3MuonIsolations = cms.EDProducer( "L3MuonCombinedRelativeIsolationProducer",
                                      UseRhoCorrectedCaloDeposits = cms.bool( False ),
                                      CaloDepositsLabel = cms.InputTag( "hltL3CaloMuonCorrectedIsolations" ),
                                      inputMuonCollection = cms.InputTag( "hltL3Muons" ),
                                      OutputMuIsoDeposits = cms.bool( True ),
                                      TrackPt_Min = cms.double( -1.0 ),
                                      printDebug = cms.bool(False),
                                      CutsPSet = cms.PSet(
    ConeSizes = cms.vdouble( 0.24 ),
    ComponentName = cms.string( "SimpleCuts" ),
    Thresholds = cms.vdouble( 0.1 ),
    maxNTracks = cms.int32( -1 ),
    EtaBounds = cms.vdouble( 2.411 ),
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
