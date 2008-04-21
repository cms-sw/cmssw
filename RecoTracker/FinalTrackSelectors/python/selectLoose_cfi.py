import FWCore.ParameterSet.Config as cms

selectLoose = cms.EDFilter("AnalyticalTrackSelector",
    src = cms.InputTag("generalTracks"),
    keepAllTracks = cms.bool(False), ## if set to true tracks failing this filter are kept in the output

    beamspot = cms.InputTag("offlineBeamSpot"),
    vtxTracks = cms.uint32(3), ## at least 3 tracks

    vtxChi2Prob = cms.double(0.01), ## at least 1% chi2nprobability (if it has a chi2)

    #untracked bool copyTrajectories = true // when doing retracking before
    copyTrajectories = cms.untracked.bool(False),
    vertices = cms.InputTag("pixelVertices"),
    qualityBit = cms.string('loose'), ## set to '' or comment out if you don't want to set the bit

    vtxNumber = cms.int32(-1),
    copyExtras = cms.untracked.bool(True), ## set to false on AOD

    minNumberLayers = cms.uint32(0),
    # parameters for cuts: loose
    chi2n_par = cms.double(2.0),
    d0_par2 = cms.vdouble(0.55, 4.0),
    d0_par1 = cms.vdouble(0.55, 4.0),
    dz_par1 = cms.vdouble(0.65, 4.0),
    # resolution parameters: normal 
    res_par = cms.vdouble(0.003, 0.01),
    dz_par2 = cms.vdouble(0.45, 4.0)
)


