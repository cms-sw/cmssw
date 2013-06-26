import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(11),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.5),
        MinE = cms.double(999.99),
        MinPhi = cms.double(-3.14159265359), ## in radians

        MaxE = cms.double(1000.01)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single electron E 1000'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)
