import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomPtAndDxyGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(-13),
        MinPt  = cms.double(2.00),
        MaxPt  = cms.double(50.00),
        MinEta = cms.double(-3.0),
        MaxEta = cms.double(3.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        LxyMax = cms.double(3000.0),#make sure most muons generated before Muon system, Gauss distribution
        LzMax = cms.double(5000.0),#make sure most muons generated before Muon system, Gauss distribution
	ConeRadius = cms.double(1000.0),
	ConeH = cms.double(3000.0),
	DistanceToAPEX = cms.double(2000.0),
        dxyMin = cms.double(0.0),
        dxyMax = cms.double(500.0)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
    psethack = cms.string('displaced muon'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)
