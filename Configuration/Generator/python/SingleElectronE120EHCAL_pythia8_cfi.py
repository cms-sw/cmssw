import FWCore.ParameterSet.Config as cms
generator = cms.EDFilter("Pythia8EGun",
                         PGunParameters = cms.PSet(
        ParticleID = cms.vint32(11),
        AddAntiParticle = cms.bool(True),
        MinPhi = cms.double(-3.14159265359), ## in radians
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-5.0),
        MaxEta = cms.double(5.0),
        MinE = cms.double(119.99),
        MaxE = cms.double(120.01)
        ),
                         Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
                         psethack = cms.string('single electron E 120 EHCAL'),
                         firstRun = cms.untracked.uint32(1),
                         PythiaParameters = cms.PSet(parameterSets = cms.vstring())
                         )
