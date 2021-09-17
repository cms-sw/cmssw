import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("Pythia8PtGun",
                         PGunParameters = cms.PSet(
        MaxPt = cms.double(150.),
        MinPt = cms.double(8.),
        ParticleID = cms.vint32(22),
        AddAntiParticle = cms.bool(True), # false in pythia6 version but photons are their own anti-particle
        MaxEta = cms.double(3.1),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-3.1),
        MinPhi = cms.double(-3.14159265359) ## in radians
        ),
                         Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
                         psethack = cms.string('single gamma pt 8 to 150'),
                         firstRun = cms.untracked.uint32(1),
                         PythiaParameters = cms.PSet(parameterSets = cms.vstring())
                         )
