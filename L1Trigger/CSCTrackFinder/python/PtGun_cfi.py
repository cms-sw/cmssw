import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(100.01),
        MinPt = cms.double(1.99),
        PartID = cms.vint32(13),
        MaxEta = cms.double(0.9),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(2.4),
        MinPhi = cms.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)
