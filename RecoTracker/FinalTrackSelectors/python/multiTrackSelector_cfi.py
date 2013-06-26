import FWCore.ParameterSet.Config as cms

looseMTS = cms.PSet(
    preFilterName=cms.string(''),
    name= cms.string('TrkLoose'),                           
    
    # vertex selection 
    vtxNumber = cms.int32(-1),
    vertexCut = cms.string('ndof>=2&!isFake'),
    
    #untracked bool copyTrajectories = true // when doing retracking before
    copyTrajectories = cms.untracked.bool(False),
    copyExtras = cms.untracked.bool(True), ## set to false on AOD
    qualityBit = cms.string('loose'), ## set to '' or comment out if you dont want to set the bit
    
    # parameters for adapted optimal cuts on chi2 and primary vertex compatibility
    chi2n_par = cms.double(1.6),
    chi2n_no1Dmod_par = cms.double(9999),
    res_par = cms.vdouble(0.003, 0.01),
    d0_par1 = cms.vdouble(0.55, 4.0),
    dz_par1 = cms.vdouble(0.65, 4.0),
    d0_par2 = cms.vdouble(0.55, 4.0),
    dz_par2 = cms.vdouble(0.45, 4.0),
    # Boolean indicating if adapted primary vertex compatibility cuts are to be applied.
    applyAdaptedPVCuts = cms.bool(True),
    
    # Impact parameter absolute cuts.
    max_d0 = cms.double(100.),
    max_z0 = cms.double(100.),
    nSigmaZ = cms.double(4.),
    
    # Cuts on numbers of layers with hits/3D hits/lost hits. 
    minNumberLayers = cms.uint32(0),
    minNumber3DLayers = cms.uint32(0),
    maxNumberLostLayers = cms.uint32(999),
    minHitsToBypassChecks = cms.uint32(20),

    # Absolute cuts in case of no PV. If yes, please define also max_d0NoPV and max_z0NoPV 
    applyAbsCutsIfNoPV = cms.bool(False),
    keepAllTracks= cms.bool(False),

    # parameters for cutting on pterror/pt and number of valid hits
    max_relpterr = cms.double(9999.),
    min_nhits = cms.uint32(0),

    max_minMissHitOutOrIn = cms.int32(99),
    max_lostHitFraction = cms.double(1.0),

    # parameters for cutting on eta
    min_eta = cms.double(-9999.),
    max_eta = cms.double(9999.)

    ) # end of pset

tightMTS=looseMTS.clone(
    preFilterName='TrkLoose',
    d0_par1 = cms.vdouble(0.3, 4.0),
    dz_par1 = cms.vdouble(0.35,4.0),
    d0_par2 = cms.vdouble(0.4, 4.0),
    dz_par2 = cms.vdouble(0.4, 4.0),
    chi2n_par = cms.double(0.7),
    chi2n_no1Dmod_par = cms.double(9999),
    name= cms.string('TrkTight'),
    minNumberLayers = cms.uint32(3),
    minNumber3DLayers = cms.uint32(3),
    maxNumberLostLayers = cms.uint32(2),
    qualityBit = cms.string('tight'), ## set to '' or comment out if you dont want to set the bit
    keepAllTracks= cms.bool(True)
    )

highpurityMTS= tightMTS.clone(
    name= cms.string('TrkHighPurity'),                           
    preFilterName='TrkTight',
    res_par=cms.vdouble(0.003,0.001),
    qualityBit = cms.string('highPurity') ## set to '' or comment out if you dont want to set the bit
)

#typical configuration is six selectors... something like this to
#make cloning easier.
multiTrackSelector = cms.EDProducer("MultiTrackSelector",
   src = cms.InputTag("generalTracks"),
   beamspot = cms.InputTag("offlineBeamSpot"),
   useVertices = cms.bool(True),
   useVtxError = cms.bool(False),
   vertices    = cms.InputTag("pixelVertices"),
   trackSelectors = cms.VPSet( looseMTS,
                               tightMTS,
                               highpurityMTS)
 ) 
