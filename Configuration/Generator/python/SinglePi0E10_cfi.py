import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(111),
        MaxEta = cms.double(3.0),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-3.0),
        MinE = cms.double(9.99),
        MinPhi = cms.double(-3.14159265359), ## in radians

        MaxE = cms.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single pi0 E 10'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)
