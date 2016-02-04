import FWCore.ParameterSet.Config as cms

bfilter = cms.EDFilter("MCSingleParticleFilter",
    MaxEta = cms.untracked.vdouble(20.0, 20.0),
    MinEta = cms.untracked.vdouble(-20.0, -20.0),
    MinPt = cms.untracked.vdouble(0.0, 0.0),
    ParticleID = cms.untracked.vint32(5, -5)
)

jpsifilter = cms.EDFilter("PythiaFilter",
    Status = cms.untracked.int32(2),
    MaxEta = cms.untracked.double(20.0),
    MinEta = cms.untracked.double(-20.0),
    MinPt = cms.untracked.double(0.0),
    ParticleID = cms.untracked.int32(443)
)

mumufilter = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(1, 1),
    MinPt = cms.untracked.vdouble(2.0, 2.0),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(-1),
    MaxInvMass = cms.untracked.double(4.0),
    MinInvMass = cms.untracked.double(2.0),
    ParticleID1 = cms.untracked.vint32(13),
    ParticleID2 = cms.untracked.vint32(13)
)
