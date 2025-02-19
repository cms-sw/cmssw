import FWCore.ParameterSet.Config as cms

L3MuonIsolations = cms.EDProducer("L3MuonIsolationProducer",
    inputMuonCollection = cms.InputTag("L3Muons"),
    CutsPSet = cms.PSet(
        ConeSizes = cms.vdouble(0.24),
        ComponentName = cms.string('SimpleCuts'),
        Thresholds = cms.vdouble(2.0),
        maxNTracks = cms.int32(-1),
        EtaBounds = cms.vdouble(999.9),
        applyCutsORmaxNTracks = cms.bool(False)
    ),
    #make it tighter if desired (if(tk.pt>Pt_Min) sumPt += tk.pt;)
    #applied after MuIsoDeposits are extracted
    TrackPt_Min = cms.double(-1.0),
    # Uncomment to set the cut based on nominal efficiency
    # lookup table
    #      PSet CutsPSet = {
    #        string ComponentName = "L3NominalEfficiencyCuts_PXLS"
    #        string LumiOption = "2E33"
    #        double NominalEfficiency = 0.97
    #      }
    OutputMuIsoDeposits = cms.bool(False),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("pixelTracks"),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string('PXLS'),
        BeamlineOption = cms.string('BeamSpotFromEvent')
    )
)



