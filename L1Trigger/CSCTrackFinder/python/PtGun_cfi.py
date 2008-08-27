import FWCore.ParameterSet.Config as cms

source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(2.01),
        MinPt = cms.untracked.double(1.51),
        PartID = cms.untracked.vint32(13),
        MaxEta = cms.untracked.double(1.8),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(0.9),
        MinPhi = cms.untracked.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single mu pt 100'),
    AddAntiParticle = cms.untracked.bool(False),
    firstRun = cms.untracked.uint32(1)
)
