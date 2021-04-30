import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomPtAndDxyGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(-13),
        MinPt = cms.double(2.0),
        MaxPt = cms.double(100.0),
        MinEta = cms.double(-2.6),
        MaxEta = cms.double(2.6),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        LxyMax = cms.double(300.0),
        LzMax = cms.double(300.0),
        ConeRadius = cms.double(1000.0),
        ConeH = cms.double(3000.0),
        DistanceToAPEX = cms.double(100.0),
        dxyMax = cms.double(100.0),
        dxyMin = cms.double(0.0)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('displaced muon'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)
