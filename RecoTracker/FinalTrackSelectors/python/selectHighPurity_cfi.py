import FWCore.ParameterSet.Config as cms

selectHighPurity = cms.EDProducer("AnalyticalTrackSelector",
    src = cms.InputTag("generalTracks"),
    keepAllTracks = cms.bool(False), ## if set to true tracks failing this filter are kept in the output
    beamspot = cms.InputTag("offlineBeamSpot"),

    # vertex selection 
    useVertices = cms.bool(True),
    useVtxError = cms.bool(False),
    vertices = cms.InputTag("pixelVertices"),
    vtxNumber = cms.int32(-1),
    vertexCut = cms.string('ndof>=2&!isFake'),

    #untracked bool copyTrajectories = true // when doing retracking before
    copyTrajectories = cms.untracked.bool(False),
    copyExtras = cms.untracked.bool(True), ## set to false on AOD
    qualityBit = cms.string('highPurity'), ## set to '' or comment out if you don't want to set the bit

    # parameters for adapted optimal cuts on chi2 and primary vertex compatibility
    chi2n_par = cms.double(0.7),
    chi2n_no1Dmod_par = cms.double(9999.),
    res_par = cms.vdouble(0.003, 0.001),
    d0_par1 = cms.vdouble(0.3, 4.0),
    dz_par1 = cms.vdouble(0.35, 4.0),
    d0_par2 = cms.vdouble(0.4, 4.0),
    dz_par2 = cms.vdouble(0.4, 4.0),
    # Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
    applyAdaptedPVCuts = cms.bool(True),

    # Impact parameter absolute cuts.
    max_d0 = cms.double(100.),
    max_z0 = cms.double(100.),
    nSigmaZ = cms.double(4.),

    # Cuts on numbers of layers with hits/3D hits/lost hits. 
    minNumberLayers = cms.uint32(3),
    minNumber3DLayers = cms.uint32(3),
    maxNumberLostLayers = cms.uint32(2),
    minHitsToBypassChecks = cms.uint32(20),

    # Absolute cuts in case of no PV. If yes, please define also max_d0NoPV and max_z0NoPV 
    applyAbsCutsIfNoPV = cms.bool(False),
    max_d0NoPV = cms.double( 100.0 ),
    max_z0NoPV = cms.double( 100.0 ),

    # parameters for cutting on pterror/pt and number of valid hits
    max_relpterr = cms.double(9999.),
    min_nhits = cms.uint32(0),

    max_minMissHitOutOrIn = cms.int32(99),
    max_lostHitFraction = cms.double(1.0),

    # parameters for cutting on eta
    max_eta = cms.double(9999.),
    min_eta = cms.double(-9999.)
                                  
)
