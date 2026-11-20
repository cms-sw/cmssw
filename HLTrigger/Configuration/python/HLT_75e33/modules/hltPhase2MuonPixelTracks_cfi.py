import FWCore.ParameterSet.Config as cms

hltPhase2MuonPixelTracks = cms.EDProducer('MuonTracksSelectorFromL1TkMuon',
    L1TkMuonInputCollection = cms.InputTag('l1tTkMuonsGmt'),
    L1TkMuMinPt = cms.double(0),
    TrackInputCollection = cms.InputTag('hltPhase2PixelTracks'),
    trackMaxPtForMatch = cms.double(50),
    maxChi2 = cms.double(9),
    maxDr = cms.double(0.1),
    maxDz = cms.double(1),
    maxCombinedQuality = cms.double(25),
    nTracksToKeep = cms.uint32(1)
)
