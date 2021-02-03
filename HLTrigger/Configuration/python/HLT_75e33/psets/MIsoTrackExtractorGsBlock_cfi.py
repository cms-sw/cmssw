import FWCore.ParameterSet.Config as cms

MIsoTrackExtractorGsBlock = cms.PSet(
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    BeamlineOption = cms.string('BeamSpotFromEvent'),
    Chi2Ndof_Max = cms.double(1e+64),
    Chi2Prob_Min = cms.double(-1.0),
    ComponentName = cms.string('TrackExtractor'),
    DR_Max = cms.double(0.5),
    DR_Veto = cms.double(0.01),
    DepositLabel = cms.untracked.string(''),
    Diff_r = cms.double(0.1),
    Diff_z = cms.double(0.2),
    NHits_Min = cms.uint32(0),
    Pt_Min = cms.double(-1.0),
    inputTrackCollection = cms.InputTag("ctfGSWithMaterialTracks")
)