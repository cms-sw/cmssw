import FWCore.ParameterSet.Config as cms

L3MuonIsolationProducerPixTE = cms.EDProducer("L3MuonIsolationProducer",
    inputMuonCollection = cms.InputTag("hltL3Muons"),
    CutsPSet = cms.PSet(
        ConeSizes = cms.vdouble(0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24),
        ComponentName = cms.string('SimpleCuts'),
        Thresholds = cms.vdouble(1.1, 1.1, 1.1, 1.1, 1.2, 
            1.1, 1.2, 1.1, 1.2, 1.0, 
            1.1, 1.0, 1.0, 1.1, 1.0, 
            1.0, 1.1, 0.9, 1.1, 0.9, 
            1.1, 1.0, 1.0, 0.9, 0.8, 
            0.1),
        maxNTracks = cms.int32(-1),
        EtaBounds = cms.vdouble(0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 
            0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 
            0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 
            1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 
            1.785, 1.88, 1.9865, 2.1075, 2.247, 
            2.411),
        applyCutsORmaxNTracks = cms.bool(False)
    ),
    TrackPt_Min = cms.double(-1.0),
    OutputMuIsoDeposits = cms.bool(True),
    ExtractorPSet = cms.PSet(
        DR_VetoPt = cms.double(0.025),
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("hltPixelTracks"),
        ReferenceRadius = cms.double(6.0),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('PixelTrackExtractor'),
        DR_Max = cms.double(0.24),
        Diff_r = cms.double(0.1),
        PropagateTracksToRadius = cms.bool(True),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string('PXLS'),
        BeamlineOption = cms.string('BeamSpotFromEvent'),
        VetoLeadingTrack = cms.bool(True),
        PtVeto_Min = cms.double(2.0)
    )
)


