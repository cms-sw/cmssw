import FWCore.ParameterSet.Config as cms

source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(50.0),
        MinPt = cms.untracked.double(50.0),
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(4.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-4.5),
        MinPhi = cms.untracked.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1)
)


