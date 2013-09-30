import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(40.01),
        MinPt = cms.double(4.99),
        PartID = cms.vint32(-12,-14,-16),
        MaxEta = cms.double(2.2),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.2),
        MinPhi = cms.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single mu pt 5-40 forward'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)
