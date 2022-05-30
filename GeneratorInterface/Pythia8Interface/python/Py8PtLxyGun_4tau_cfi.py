import FWCore.ParameterSet.Config as cms

#Note: distances in mm instead of in cm usually used in CMS
generator = cms.EDFilter("Pythia8PtAndLxyGun",

    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),

    PGunParameters = cms.PSet(
        ParticleID = cms.vint32(-15, -15),
        AddAntiParticle = cms.bool(True), # antiparticle has opposite momentum and production point symmetric wrt (0,0,0) compared to corresponding particle
        MinPt  = cms.double(15.00),
        MaxPt  = cms.double(300.00),
        MinEta = cms.double(-2.5),
        MaxEta = cms.double(2.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        LxyMin = cms.double(0.0),
        LxyMax = cms.double(550.0), # most tau generated within TOB (55cm)
        LzMax = cms.double(300.0),
        dxyMax = cms.double(30.0),
        dzMax = cms.double(120.0),
        ConeRadius = cms.double(1000.0),
        ConeH = cms.double(3000.0),
        DistanceToAPEX = cms.double(850.0),
        LxyBackFraction = cms.double(0.0), # fraction of particles going back towards to center at transverse plan; numbers outside the [0,1] range are set to 0 or 1
        LzOppositeFraction = cms.double(0.0), # fraction of particles going in opposite direction wrt to center along beam-line than in transverse plane; numbers outside the [0,1] range are set to 0 or 1
    ),

    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
    psethack = cms.string('displaced taus'),
    firstRun = cms.untracked.uint32(1),
    PythiaParameters = cms.PSet(parameterSets = cms.vstring())
)
