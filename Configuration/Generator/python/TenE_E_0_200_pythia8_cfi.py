import FWCore.ParameterSet.Config as cms

# Modified from Configuration/Generator/python/TenMuE_0_200_pythia8_cfi.py
generator = cms.EDFilter("Pythia8EGun",
    PGunParameters = cms.PSet(
        MaxE = cms.double(200.0),
        MinE = cms.double(0.0),
        ParticleID = cms.vint32(-11,-11,-11,-11,-11),
        AddAntiParticle = cms.bool(True),
        MaxEta = cms.double(3.),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-3.),
        MinPhi = cms.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('Ten e w/ energy 0 to 200'),
    firstRun = cms.untracked.uint32(1),
    PythiaParameters = cms.PSet(parameterSets = cms.vstring())

)
