import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(11),
        MinPhi = cms.double(-3.14159265359), ## in radians
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-5.0),
        MaxEta = cms.double(5.0),
        MinE = cms.double(119.99),
        MaxE = cms.double(120.01)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single electron E 120 EHCAL'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)
