import FWCore.ParameterSet.Config as cms

#loose
hiLooseMTS = cms.PSet(
    preFilterName=cms.string(''),
    name= cms.string('hiTrkLoose'),

   # vertex selection
    vtxNumber = cms.int32(-1),
    vertexCut = cms.string(''),

    #untracked bool copyTrajectories = true // when doing retracking before
    copyTrajectories = cms.untracked.bool(True),
    copyExtras = cms.untracked.bool(True), ## set to false on AOD
    qualityBit = cms.string('loose'), ## set to '' or comment out if you dont want to set the

    chi2n_par = cms.double(9999),                     # version with 1D hits modification
    chi2n_no1Dmod_par = cms.double(0.3),                     # normalizedChi2 < nLayers * chi2n_par
    res_par = cms.vdouble(99999., 99999.),            # residual parameterization (re-check in HI)
    d0_par1 = cms.vdouble(9999., 0.),                 # parameterized nomd0E
    dz_par1 = cms.vdouble(9999., 0.),
    d0_par2 = cms.vdouble(8.0, 0.0),              # d0E from tk.d0Error
    dz_par2 = cms.vdouble(8.0, 0.0),
    # Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
    applyAdaptedPVCuts = cms.bool(True),

    # Impact parameter absolute cuts.
    max_z0 = cms.double(1000),
    max_d0 = cms.double(1000),
    nSigmaZ = cms.double(9999.),

   # Cuts on numbers of layers with hits/3D hits/lost hits.
    minNumberLayers = cms.uint32(0),
    minNumber3DLayers = cms.uint32(0),
    maxNumberLostLayers = cms.uint32(999),
    minHitsToBypassChecks = cms.uint32(20),
    max_minMissHitOutOrIn = cms.int32(99),
    max_lostHitFraction = cms.double(1.0),
    min_eta = cms.double(-9999.),
    max_eta = cms.double(9999.) ,

    # Absolute cuts in case of no PV. If yes, please define also max_d0NoPV and max_z0NoPV
    applyAbsCutsIfNoPV = cms.bool(False),
    keepAllTracks= cms.bool(False),

    # parameters for cutting on pterror/pt and number of valid hits
    max_relpterr = cms.double(0.1),
    min_nhits = cms.uint32(10)
    )

hiTightMTS=hiLooseMTS.clone(
    preFilterName='hiTrkLoose',
    min_nhits = cms.uint32(12),
    max_relpterr = cms.double(0.075),
    d0_par2 = cms.vdouble(5.0, 0.0),
    dz_par2 = cms.vdouble(5.0, 0.0),
    chi2n_no1Dmod_par = cms.double(0.25),
    name= cms.string('hiTrkTight'),
    qualityBit = cms.string('tight'), ## set to '' or comment out if you dont want to set the bit
    keepAllTracks= cms.bool(True)
    )

hiHighpurityMTS= hiTightMTS.clone(
    name= cms.string('hiTrkHighPurity'),
    preFilterName='hiTrkTight',
    min_nhits = cms.uint32(13),
    max_relpterr = cms.double(0.05),
    d0_par2 = [3.0, 0.0],
    dz_par2 = [3.0, 0.0],
    chi2n_no1Dmod_par = cms.double(0.15),
    qualityBit = cms.string('highPurity') ## set to '' or comment out if you dont want to set the bit
    )

#typical configuration is six selectors... something like this to
#make cloning easier.
hiMultiTrackSelector = cms.EDProducer("MultiTrackSelector",
                                    src = cms.InputTag("hiGeneralTracks"),
                                    beamspot = cms.InputTag("offlineBeamSpot"),
                                    useVertices = cms.bool(True),
                                    useVtxError = cms.bool(True),
                                    vertices    = cms.InputTag("hiSelectedVertex"),
                                    trackSelectors = cms.VPSet( hiLooseMTS,
                                                                hiTightMTS,
                                                                hiHighpurityMTS)
                                    )
