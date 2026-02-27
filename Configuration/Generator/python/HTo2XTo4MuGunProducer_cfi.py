import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("HTo2XTo4LGunProducer",
    PGunParameters = cms.PSet(
        PartID   = cms.vint32(-13),
        MinMassH = cms.double(1.00),
        MaxMassH = cms.double(1000.00),
        MinPtH   = cms.double(1.00),
        MaxPtH   = cms.double(120.00),
        MinEta   = cms.double(-3.5),
        MaxEta   = cms.double(3.5),
        MinPhi   = cms.double(-3.14159265359),
        MaxPhi   = cms.double(3.14159265359),
        MaxCTauLLP = cms.double(5000),
        MinCTauLLP = cms.double(0.01),
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
    psethack = cms.string('H -> 2X -> 4mu'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)
