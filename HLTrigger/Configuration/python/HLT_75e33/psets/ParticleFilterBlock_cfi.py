import FWCore.ParameterSet.Config as cms

ParticleFilterBlock = cms.PSet(
    ParticleFilter = cms.PSet(
        EMin = cms.double(0.1),
        chargedPtMin = cms.double(0.1),
        etaMax = cms.double(5.3),
        invisibleParticles = cms.vint32(),
        protonEMin = cms.double(5000.0),
        rMax = cms.double(129.0),
        zMax = cms.double(317.0)
    )
)