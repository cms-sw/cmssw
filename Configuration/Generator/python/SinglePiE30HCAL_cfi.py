import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MaxEta = cms.double(5.0),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-5.0),
        MinE = cms.double(29.99),
        MinPhi = cms.double(-3.14159265359), ## in radians

        MaxE = cms.double(30.01)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single pi E 30 HCAL'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)
