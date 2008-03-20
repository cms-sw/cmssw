import FWCore.ParameterSet.Config as cms

source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(22),
        MaxEta = cms.untracked.double(5.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-5.0),
        MinE = cms.untracked.double(119.99),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxE = cms.untracked.double(120.01)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single gamma E 120 EHCAL'),
    firstRun = cms.untracked.uint32(1)
)



