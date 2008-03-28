import FWCore.ParameterSet.Config as cms

source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(10.0),
        MinPt = cms.untracked.double(10.0),
        # you can request more than 1 particle
        #untracked vint32  PartID = { 211, 11, -13 }
        PartID = cms.untracked.vint32(13),
        MaxEta = cms.untracked.double(3.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-3.0),
        MinPhi = cms.untracked.double(-3.14159265359) ## it must be in radians

    ),
    Verbosity = cms.untracked.int32(0), ## for printouts, set it to 1 (or greater)   

    firstRun = cms.untracked.uint32(1)
)


