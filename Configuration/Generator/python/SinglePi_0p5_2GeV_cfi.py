import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(2.0),
        MinPt = cms.double(0.5),
        MaxEta = cms.double(3.0),
        MinEta = cms.double(-3.0),
        MaxPhi = cms.double(3.14159265359),
        MinPhi = cms.double(-3.14159265359),
        AddAntiParticle = cms.bool(True),
        ParticleID = cms.vint32(211)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single pi pt 0.5 to 2'),
    firstRun = cms.untracked.uint32(1),
    PythiaParameters = cms.PSet(parameterSets = cms.vstring())
)
