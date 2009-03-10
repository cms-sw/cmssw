import FWCore.ParameterSet.Config as cms

source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(11),
        MinPhi = cms.untracked.double(-3.14159265359), ## in radians
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-5.0),
        MaxEta = cms.untracked.double(5.0),
        MinE = cms.untracked.double(119.99),
        MaxE = cms.untracked.double(120.01)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single electron E 120 EHCAL'),
    firstRun = cms.untracked.uint32(1)
)


