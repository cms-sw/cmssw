import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(1.0),
        MinPt = cms.double(0.3),
        ParticleID = cms.vint32(11),
        AddAntiParticle = cms.bool(True),
        MaxEta = cms.double(4.5),
        MinEta = cms.double(-4.5),
        MaxPhi = cms.double(3.14159265359),
        MinPhi = cms.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single electron pt 0.3 to 1.0'),
    firstRun = cms.untracked.uint32(1),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    )
)
