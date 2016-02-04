import FWCore.ParameterSet.Config as cms

MuMuFilter = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(1, 1),
    MinPt = cms.untracked.vdouble(2.5, 2.5),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(-1),
    MaxInvMass = cms.untracked.double(5.5),
    MinInvMass = cms.untracked.double(5.3),
    ParticleID1 = cms.untracked.vint32(13),
    ParticleID2 = cms.untracked.vint32(13)
)

# -- Require Muon from Bs
MuFilter = cms.EDFilter("PythiaFilter",
    Status = cms.untracked.int32(1),
    MotherID = cms.untracked.int32(531),
    MinPt = cms.untracked.double(2.5),
    ParticleID = cms.untracked.int32(13),
    MaxEta = cms.untracked.double(2.5),
    MinEta = cms.untracked.double(-2.5)
)
