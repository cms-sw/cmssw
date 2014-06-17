import FWCore.ParameterSet.Config as cms

EleIsoTrackExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaTrackExtractor'),
    inputTrackCollection = cms.InputTag("generalTracks"),
    DepositLabel = cms.untracked.string(''),
    Diff_r = cms.double(9999.0),
    Diff_z = cms.double(0.2),
    DR_Max = cms.double(1.0),
    DR_Veto = cms.double(0.0),

    BeamlineOption = cms.string('BeamSpotFromEvent'),
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    NHits_Min = cms.uint32(0),
    Chi2Ndof_Max = cms.double(1e+64),
    Chi2Prob_Min = cms.double(-1.0),
    Pt_Min = cms.double(-1.0),

    dzOption = cms.string("vz")
)
