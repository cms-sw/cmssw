import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("Pythia8EGun",
    PGunParameters = cms.PSet(
        MaxEta = cms.double(3.0),
        MinEta = cms.double(-3.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE = cms.double(0.2),
        MaxE = cms.double(1.0),
        MinPt = cms.double(0.3),
        MaxPt = cms.double(0.5),
        AddAntiParticle = cms.bool(True),
        ParticleID = cms.vint32(211)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single pion pt 0.3 to 0.5'),
    firstRun = cms.untracked.uint32(1),
    PythiaParameters = cms.PSet(parameterSets = cms.vstring())
)
