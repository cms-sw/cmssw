import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(22),
        MinPhi = cms.double(-3.14159265359), ## in radians
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-5.0),
        MaxEta = cms.double(5.0),
        MinE = cms.double(49.99),
        MaxE = cms.double(50.01)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single gamma E 50'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)
