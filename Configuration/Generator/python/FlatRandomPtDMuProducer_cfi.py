import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomPtAndDxyGunProducer",
    PGunParameters = cms.PSet(
        ConeH = cms.double(3000.0),
        ConeRadius = cms.double(1000.0),
        DistanceToAPEX = cms.double(100.0),
        LxyMax = cms.double(300.0),
        LzMax = cms.double(300.0),
        MaxEta = cms.double(2.6),
        MaxPhi = cms.double(3.14159265359),
        MaxPt = cms.double(8.0),
        MinEta = cms.double(-2.6),
        MinPhi = cms.double(-3.14159265359),
        MinPt = cms.double(1.5),
        PartID = cms.vint32(-13),
        dxyMax = cms.double(100.0),
        dxyMin = cms.double(0.0)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    AddAntiParticle = cms.bool(True),
    psethack = cms.string('displaced muon')
)
