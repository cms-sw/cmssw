import FWCore.ParameterSet.Config as cms
generator = cms.EDFilter("Pythia8EGun",
                         PGunParameters = cms.PSet(
        ParticleID = cms.vint32(211),
        AddAntiParticle = cms.bool(True),
        MaxEta = cms.double(5.0),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-5.0),
        MinE = cms.double(49.99),
        MinPhi = cms.double(-3.14159265359), ## in radians
        MaxE = cms.double(50.01)
        ),
                         Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
                         psethack = cms.string('single pi E 50 HCAL'),
                         firstRun = cms.untracked.uint32(1),
                         PythiaParameters = cms.PSet(parameterSets = cms.vstring())
                         )

