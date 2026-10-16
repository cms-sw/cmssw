import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("Pythia8EGun",
    PGunParameters = cms.PSet(
        MaxEta = cms.double(3.0),
        MinEta = cms.double(-3.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE = cms.double(0.4),
        MaxE = cms.double(3.0),
        MinPt = cms.double(0.5),
        MaxPt = cms.double(2.0),
        AddAntiParticle = cms.bool(True),
        ParticleID = cms.vint32(11)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single electron pt 0.5 to 2'),
    firstRun = cms.untracked.uint32(1),
    PythiaParameters = cms.PSet(parameterSets = cms.vstring())
)
