import FWCore.ParameterSet.Config as cms

RU_ME0 = cms.PSet(
    allowWideSegments = cms.bool(True),
    doCollisions = cms.bool(True),
    maxChi2Additional = cms.double(100.0),
    maxChi2Prune = cms.double(50),
    maxChi2GoodSeg = cms.double(50),
    maxPhiSeeds = cms.double(0.001096605744), #Assuming 384 strips
    maxPhiAdditional = cms.double(0.001096605744), #Assuming 384 strips
    maxETASeeds = cms.double(0.1), #Assuming 8 eta partitions
    maxTOFDiff = cms.double(25),
    requireCentralBX = cms.bool(True), #require that a majority of hits come from central BX
    minNumberOfHits = cms.uint32(4),
)

ME0SegAlgoRU = cms.PSet(
    algo_name = cms.string('ME0SegAlgoRU'),
    algo_pset = cms.PSet(cms.PSet(RU_ME0))
)

