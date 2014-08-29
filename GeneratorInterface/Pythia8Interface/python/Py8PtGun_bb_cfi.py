import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter(
    "Pythia8PtGun",

    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    
    PGunParameters = cms.PSet(
        ParticleID = cms.vint32(5),
        AddAntiParticle = cms.bool(True),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinPt = cms.double(100.0),
        MaxPt = cms.double(200.0),
        MinEta = cms.double(0.0),
        MaxEta = cms.double(2.4)
        ),
    
    PythiaParameters = cms.PSet(parameterSets = cms.vstring())       
)
