import FWCore.ParameterSet.Config as cms

RU_ME0 = cms.PSet(
    allowWideSegments = cms.bool(True),
    doCollisions = cms.bool(True),
    maxChi2Additional = cms.double(100.0),
    maxChi2GoodSeg = cms.double(50),
    maxChi2Prune = cms.double(50),
    maxETASeeds = cms.double(0.1),
    maxPhiAdditional = cms.double(0.001096605744),
    maxPhiSeeds = cms.double(0.001096605744),
    maxTOFDiff = cms.double(25),
    minNumberOfHits = cms.uint32(4),
    requireCentralBX = cms.bool(True)
)