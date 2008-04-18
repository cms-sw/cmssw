import FWCore.ParameterSet.Config as cms

source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(1.01),
        MinPt = cms.untracked.double(0.99),
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(5.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-5.0),
        MinPhi = cms.untracked.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single pi pt 1'),
    AddAntiParticle = cms.untracked.bool(True),
    firstRun = cms.untracked.uint32(1)
)



