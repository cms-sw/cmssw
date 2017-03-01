import FWCore.ParameterSet.Config as cms

# Modified from Configuration/Generator/python/SingleMuPt10_cfi.py
generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(200.0),
        MinPt = cms.double(0.9),
        ParticleID = cms.vint32(-13,-13),
        AddAntiParticle = cms.bool(True),
        MaxEta = cms.double(4.0),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-4.0),
        MinPhi = cms.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('Four mu pt 1 to 200'),
    firstRun = cms.untracked.uint32(1),
    PythiaParameters = cms.PSet(parameterSets = cms.vstring())
)
