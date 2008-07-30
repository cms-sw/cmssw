import FWCore.ParameterSet.Config as cms

source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(100.01),
        MinPt = cms.untracked.double(99.99),
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(5.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-5.0),
        MinPhi = cms.untracked.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single pi pt 100'),
    AddAntiParticle = cms.untracked.bool(True),
    firstRun = cms.untracked.uint32(1)
)


