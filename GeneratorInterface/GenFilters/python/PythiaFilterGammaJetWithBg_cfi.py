import FWCore.ParameterSet.Config as cms

gj_filter = cms.EDFilter("PythiaFilterGammaJetWithBg",
    MaxEvents = cms.untracked.int32(2),
    MaxPhotonEta = cms.untracked.double(2.8),
    MaxPhotonPt = cms.untracked.double(22.0),
    MinPhotonEtaForwardJet = cms.untracked.double(1.3),
    MinDeltaPhi = cms.untracked.double(170.0),
    MinPhotonPt = cms.untracked.double(18.0),
    MaxDeltaEta = cms.untracked.double(1.3),
    PhotonSeedPt = cms.untracked.double(5.0)
)


