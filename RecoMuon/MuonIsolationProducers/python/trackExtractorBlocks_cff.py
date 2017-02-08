import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
MIsoTrackExtractorCtfBlock = cms.PSet(
    Diff_z = cms.double(0.2),
    inputTrackCollection = cms.InputTag("generalTracks"),
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    ComponentName = cms.string('TrackExtractor'),
    DR_Max = cms.double(1.0),
    Diff_r = cms.double(0.1),
    Chi2Prob_Min = cms.double(-1.0),
    DR_Veto = cms.double(0.01),
    NHits_Min = cms.uint32(0),
    Chi2Ndof_Max = cms.double(1e+64),
    Pt_Min = cms.double(-1.0),
    DepositLabel = cms.untracked.string(''),
    BeamlineOption = cms.string('BeamSpotFromEvent')
)
MIsoTrackExtractorBlock = cms.PSet(
    Diff_z = cms.double(0.2),
    inputTrackCollection = cms.InputTag("generalTracks"),
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    ComponentName = cms.string('TrackExtractor'),
    DR_Max = cms.double(1.0),
    Diff_r = cms.double(0.1),
    Chi2Prob_Min = cms.double(-1.0),
    DR_Veto = cms.double(0.01),
    NHits_Min = cms.uint32(0),
    Chi2Ndof_Max = cms.double(1e+64),
    Pt_Min = cms.double(-1.0),
    DepositLabel = cms.untracked.string(''),
    BeamlineOption = cms.string('BeamSpotFromEvent')
)
MIsoTrackExtractorGsBlock = cms.PSet(
    Diff_z = cms.double(0.2),
    inputTrackCollection = cms.InputTag("ctfGSWithMaterialTracks"),
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    ComponentName = cms.string('TrackExtractor'),
    DR_Max = cms.double(1.0),
    Diff_r = cms.double(0.1),
    Chi2Prob_Min = cms.double(-1.0),
    DR_Veto = cms.double(0.01),
    NHits_Min = cms.uint32(0),
    Chi2Ndof_Max = cms.double(1e+64),
    Pt_Min = cms.double(-1.0),
    DepositLabel = cms.untracked.string(''),
    BeamlineOption = cms.string('BeamSpotFromEvent')
)


