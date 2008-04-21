import FWCore.ParameterSet.Config as cms

selectCKF = cms.EDFilter("AnalyticalTrackSelector",
    src = cms.InputTag("generalTracks"),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vtxTracks = cms.uint32(3), ## at least 3 tracks

    vtxChi2Prob = cms.double(0.01), ## at least 1% chi2nprobability (if it has a chi2)

    #untracked bool copyTrajectories = true // when doing retracking before
    copyTrajectories = cms.untracked.bool(False),
    vertices = cms.InputTag("pixelVertices"),
    vtxNumber = cms.int32(-1),
    copyExtras = cms.untracked.bool(True), ## set to false on AOD

    minNumberLayers = cms.uint32(0),
    # parameters for cuts: tight // loose
    chi2n_par = cms.double(0.9),
    d0_par2 = cms.vdouble(0.55, 4.0), ##{ 0.40, 4. }

    d0_par1 = cms.vdouble(0.55, 4.0), ##{ 0.30, 4. }

    dz_par1 = cms.vdouble(0.65, 4.0), ##{ 0.35, 4. }

    # resolution parameters: normal // tighter
    res_par = cms.vdouble(0.003, 0.01),
    dz_par2 = cms.vdouble(0.45, 4.0) ##{ 0.40, 4. }

)


