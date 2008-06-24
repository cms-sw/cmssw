import FWCore.ParameterSet.Config as cms

source = cms.Source(
    "FlatRandomPtGunSource",
    firstRun = cms.untracked.uint32(1),
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(13),
        # you can request more than 1 particle
        # PartID = cms.untracked.vint32(211,11,-13),
        MinPt = cms.untracked.double(10.0),
        MaxPt = cms.untracked.double(10.0),
        MinEta = cms.untracked.double(-3.0),
        MaxEta = cms.untracked.double(3.0),
        MinPhi = cms.untracked.double(-3.14159265359), ## it must be in radians
        MaxPhi = cms.untracked.double(3.14159265359),
    ),
    AddAntiParticle = cms.untracked.bool(false), # back-to-back particles
    Verbosity = cms.untracked.int32(0) ## for printouts, set it to 1 (or greater)   
)


