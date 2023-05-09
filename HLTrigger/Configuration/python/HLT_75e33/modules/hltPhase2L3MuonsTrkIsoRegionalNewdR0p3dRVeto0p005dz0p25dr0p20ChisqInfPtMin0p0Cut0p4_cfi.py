import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonsTrkIsoRegionalNewdR0p3dRVeto0p005dz0p25dr0p20ChisqInfPtMin0p0Cut0p4 = cms.EDProducer("L3MuonCombinedRelativeIsolationProducer",
    CaloDepositsLabel = cms.InputTag("notUsed"),
    CaloExtractorPSet = cms.PSet(
        CaloTowerCollectionLabel = cms.InputTag("hltTowerMakerForAll"),
        ComponentName = cms.string('CaloExtractor'),
        DR_Max = cms.double(0.3),
        DR_Veto_E = cms.double(0.07),
        DR_Veto_H = cms.double(0.1),
        DepositLabel = cms.untracked.string('EcalPlusHcal'),
        Threshold_E = cms.double(0.2),
        Threshold_H = cms.double(0.5),
        Vertex_Constraint_XY = cms.bool(False),
        Vertex_Constraint_Z = cms.bool(False),
        Weight_E = cms.double(1.0),
        Weight_H = cms.double(1.0)
    ),
    CutsPSet = cms.PSet(
        ComponentName = cms.string('SimpleCuts'),
        ConeSizes = cms.vdouble(0.3),
        EtaBounds = cms.vdouble(2.411),
        Thresholds = cms.vdouble(0.4),
        applyCutsORmaxNTracks = cms.bool(False),
        maxNTracks = cms.int32(-1)
    ),
    OutputMuIsoDeposits = cms.bool(True),
    TrackPt_Min = cms.double(-1.0),
    TrkExtractorPSet = cms.PSet(
        BeamSpotLabel = cms.InputTag("hltOnlineBeamSpot"),
        BeamlineOption = cms.string('BeamSpotFromEvent'),
        Chi2Ndof_Max = cms.double(1e+64),
        Chi2Prob_Min = cms.double(-1.0),
        ComponentName = cms.string('PixelTrackExtractor'),
        DR_Max = cms.double(0.3),
        DR_Veto = cms.double(0.005),
        DR_VetoPt = cms.double(0.025),
        DepositLabel = cms.untracked.string('PXLS'),
        Diff_r = cms.double(0.2),
        Diff_z = cms.double(0.25),
        NHits_Min = cms.uint32(0),
        PropagateTracksToRadius = cms.bool(True),
        PtVeto_Min = cms.double(2.0),
        Pt_Min = cms.double(-1.0),
        ReferenceRadius = cms.double(6.0),
        VetoLeadingTrack = cms.bool(True),
        inputTrackCollection = cms.InputTag("hltPhase2L3MuonGeneralTracks")
    ),
    UseCaloIso = cms.bool(False),
    UseRhoCorrectedCaloDeposits = cms.bool(False),
    inputMuonCollection = cms.InputTag("hltPhase2L3MuonCandidates"),
    printDebug = cms.bool(False)
)
