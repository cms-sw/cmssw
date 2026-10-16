import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(2.0),
        MinPt = cms.double(0.5),
        # [211] * 50 creates a list of 50 pions. 
        # AddAntiParticle=True doubles it to 100 (50 pi+, 50 pi-)!
        ParticleID = cms.vint32([211] * 50),
        AddAntiParticle = cms.bool(True),
        MaxEta = cms.double(4.5),
        MinEta = cms.double(-4.5),
        MaxPhi = cms.double(3.14159265359),
        MinPhi = cms.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('multi pion pt 0.5 to 2.0 100 per event'),
    firstRun = cms.untracked.uint32(1),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    )
)
