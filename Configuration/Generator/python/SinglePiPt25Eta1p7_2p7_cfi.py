import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(25.01),
        MinPt = cms.double(24.99),
        PartID = cms.vint32(211),
        MaxEta = cms.double(2.7),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(1.7),
        MinPhi = cms.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single pi pt 25'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)
