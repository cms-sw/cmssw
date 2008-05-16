import FWCore.ParameterSet.Config as cms

source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(11),
        MaxEta = cms.untracked.double(2.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-2.5),
        MinE = cms.untracked.double(999.99),
        MinPhi = cms.untracked.double(-3.14159265359), ## in radians

        MaxE = cms.untracked.double(1000.01)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single electron E 1000'),
    AddAntiParticle = cms.untracked.bool(True),
    firstRun = cms.untracked.uint32(1)
)



