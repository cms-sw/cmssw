import FWCore.ParameterSet.Config as cms

isotrack_filter = cms.EDFilter("PythiaFilterIsolatedTrack",
    MaxPhotonEta = cms.untracked.double(2.3),
    PhotonSeedPt = cms.untracked.double(10.0),
    isoCone = cms.untracked.double(0.5)
)


