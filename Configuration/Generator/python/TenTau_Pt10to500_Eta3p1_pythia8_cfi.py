import FWCore.ParameterSet.Config as cms

# Modified from Configuration/Generator/python/TenMuE_0_200_pythia8_cfi.py
generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        ParticleID = cms.vint32(-15,-15,-15,-15,-15),
        AddAntiParticle = cms.bool(True),
        MinPt = cms.double(10.),
        MaxPt = cms.double(500.),
        MinEta = cms.double(-3.1),
        MaxEta = cms.double(3.1),
        MinPhi = cms.double(-3.14159265359), # in radians
        MaxPhi = cms.double(3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0), # set to 1 (or greater)  for printouts
    psethack = cms.string("Ten tau leptons w/ uniform 10 < pT < 500 GeV, |eta| < 3.1"),
    firstRun = cms.untracked.uint32(1),
    PythiaParameters = cms.PSet(parameterSets = cms.vstring())
)
