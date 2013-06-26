import FWCore.ParameterSet.Config as cms

btagGenBb = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(2, 2),
    MinPt = cms.untracked.vdouble(10.0, 10.0),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(-1),
    ParticleID1 = cms.untracked.vint32(5),
    ParticleID2 = cms.untracked.vint32(5)
)


