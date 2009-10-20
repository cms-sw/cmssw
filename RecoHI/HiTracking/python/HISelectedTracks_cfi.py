import FWCore.ParameterSet.Config as cms

hiSelectedTracks = cms.EDProducer("AnalyticalTrackSelector",
    src = cms.InputTag("hiGlobalPrimTracks"),
    keepAllTracks = cms.bool(False), ## if set to true tracks failing this filter are kept in the output
    beamspot = cms.InputTag("offlineBeamSpot"),
                           
    vertices = cms.InputTag("hiSelectedVertex"),
    vtxNumber = cms.int32(-1),
    vtxTracks = cms.uint32(0), ## at least 3 tracks by default
    vtxChi2Prob = cms.double(0.01), ## at least 1% chi2nprobability (if it has a chi2)

    copyTrajectories = cms.untracked.bool(True),  ## false by default, needed by TrackClusterRemover
    copyExtras = cms.untracked.bool(True), ## set to false on AOD
    qualityBit = cms.string(''), ## set to '' or comment out if you don't want to set the bit

    # parameters for adapted optimal cuts on chi2 and primary vertex compatibility
    chi2n_par = cms.double(0.4),         # normalizedChi2 < nLayers * chi2n_par
    res_par = cms.vdouble(0.003, 0.001), # residual parameterization should be re-checked in HI
    d0_par1 = cms.vdouble(9999, 1),      # parameterized nomd0E
    dz_par1 = cms.vdouble(9999, 1),
    d0_par2 = cms.vdouble(5.0, 0.3),     # d0E from tk.d0Error
    dz_par2 = cms.vdouble(30.0, 0.3),
    # Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
    applyAdaptedPVCuts = cms.bool(True),

    # Impact parameter absolute cuts.
    max_d0 = cms.double(100.),
    max_z0 = cms.double(100.),

    # Cuts on numbers of layers with hits/3D hits/lost hits. 
    minNumberLayers = cms.uint32(7),
    minNumber3DLayers = cms.uint32(3),
    maxNumberLostLayers = cms.uint32(999)
 
)
