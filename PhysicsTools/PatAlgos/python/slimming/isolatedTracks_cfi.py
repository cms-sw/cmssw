import FWCore.ParameterSet.Config as cms

isolatedTracks = cms.EDProducer("PATIsolatedTrackProducer",
    packedCandidates = cms.InputTag("packedPFCandidates"),
    pT_cut = cms.double(5.0),
    dR_cut = cms.double(0.3),
    dZ_cut = cms.double(0.1),
    absIso_cut = cms.double(5.0),
    relIso_cut = cms.double(0.2),
    miniRelIso_cut = cms.double(0.2),
)
