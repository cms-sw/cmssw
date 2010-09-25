import FWCore.ParameterSet.Config as cms


selectTracks = cms.EDFilter("RecoTrackSelector",
    src = cms.InputTag("generalTracks"),
    maxChi2 = cms.double(10000.0),
    tip = cms.double(3),                     #d0/sigmad0
    minRapidity = cms.double(-2.5),
    lip = cms.double(3),                     #dz/sigmadz
    ptMin = cms.double(0.5),
    maxRapidity = cms.double(2.5),
    vtxntk = cms.double(3),                  #selection evets with vertex recostructed with at least 3 tracks non attivo
    ndof= cms.double(4),
    pxlLayerMinCut = cms.double(0),          #selection tracks with at least 2 hits in the pixel
    requirePIX1 = cms.bool(False),           #selection tracks with a hit a first layer of the pixel barrel or endcap 
    quality = cms.vstring('highPurity'),
    algorithm = cms.vstring(),
    minHit = cms.int32(3),                   #selection tracks with at least 4 layers crossed
    min3DHit = cms.int32(0),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    vertexCollection = cms.InputTag("offlinePrimaryVertices"),
    diffvtxbs =cms.double(10.),              #change selection from 10 to 15 
    ptErr_pt = cms.double(0.05),                #5%  
     bsuse = cms.bool(False),
    allowTriplets = cms.bool(False)            #change UE3 selection
	
)



goodTracks= cms.EDProducer("ChargedRefCandidateProducer",
    src = cms.InputTag("selectTracks"),
#    src = cms.InputTag("generalTracks"),
     particleType = cms.string('pi+')
)




#goodTracks = cms.EDFilter("PtMinCandViewSelector",
#    src = cms.InputTag("allTracks"),
#    ptMin = cms.double(0.290)
#)


UEAnalysisTracks = cms.Sequence(selectTracks*goodTracks)


