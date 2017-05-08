import FWCore.ParameterSet.Config as cms

stoppedTracks = cms.EDProducer("PATStoppedTrackProducer",
    packedPFCandidates = cms.InputTag("packedPFCandidates"),
    lostTracks = cms.InputTag("lostTracks"),
    generalTracks = cms.InputTag("generalTracks"),
    dEdxInfo = cms.InputTag("dedxHarmonic2"),
    dEdxHitInfo = cms.InputTag("dedxHitInfo"),
    pT_cut = cms.double(20.0),
    dR_cut = cms.double(0.3),
    dZ_cut = cms.double(0.1),
    absIso_cut = cms.double(5.0),
    relIso_cut = cms.double(0.2),
    miniRelIso_cut = cms.double(0.2),
)
