import FWCore.ParameterSet.Config as cms

muGenFilter = cms.EDFilter("MCSmartSingleParticleFilter",
    MaxDecayRadius = cms.untracked.vdouble(2000.0, 2000.0),
    MaxDecayZ = cms.untracked.vdouble(4000.0, 4000.0),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    MinDecayZ = cms.untracked.vdouble(-4000.0, -4000.0),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    MinPt = cms.untracked.vdouble(5.0, 5.0),
    ParticleID = cms.untracked.vint32(13, -13),
    Status = cms.untracked.vint32(1, 1),
    moduleLabel = cms.untracked.InputTag("generatorSmeared","","SIM")
)
