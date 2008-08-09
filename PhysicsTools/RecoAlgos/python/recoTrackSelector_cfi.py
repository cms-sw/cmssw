import FWCore.ParameterSet.Config as cms

recoTrackSelector = cms.EDFilter("RecoTrackSelector",
    src = cms.InputTag("generalTracks"),
    maxChi2 = cms.double(10000.0),
    tip = cms.double(120.0),
    minRapidity = cms.double(-5.0),
    lip = cms.double(300.0),
    ptMin = cms.double(0.1),
    maxRapidity = cms.double(5.0),
    #enum defined in DataFormats/TrackReco/interface/TrackBase.h
    #enum TrackQuality {undefQuality=-1,loose=0,tight=1,highPurity=2,confirmed=3,goodIterative=4};
    quality = cms.vint32(0),
    #enum TrackAlgorithm {undefAlgorithm=0,ctf=1,rs=2,cosmics=3,beamhalo=4,iter1=5,iter2=6,iter3=7};
    algorithm = cms.vint32(),
    minHit = cms.int32(3),
    beamSpot = cms.InputTag("offlineBeamSpot")
)



