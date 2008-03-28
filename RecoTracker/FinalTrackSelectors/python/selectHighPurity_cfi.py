import FWCore.ParameterSet.Config as cms

selectHighPurity = cms.EDFilter("AnalyticalTrackSelector",
    src = cms.InputTag("generalTracks"),
    keepAllTracks = cms.bool(False),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vtxTracks = cms.uint32(3),
    vtxChi2Prob = cms.double(0.01),
    #untracked bool copyTrajectories = true // when doing retracking before
    copyTrajectories = cms.untracked.bool(False),
    vertices = cms.InputTag("pixelVertices"),
    qualityBit = cms.string('highPurity'),
    vtxNumber = cms.int32(-1),
    copyExtras = cms.untracked.bool(True),
    minNumberLayers = cms.uint32(5),
    # parameters for cuts: tight 
    chi2n_par = cms.double(0.9),
    d0_par2 = cms.vdouble(0.4, 4.0),
    d0_par1 = cms.vdouble(0.3, 4.0),
    dz_par1 = cms.vdouble(0.35, 4.0),
    dz_par2 = cms.vdouble(0.4, 4.0)
)


