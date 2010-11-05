import FWCore.ParameterSet.Config as cms

HBHENoiseFilter = cms.EDFilter(
    'HBHENoiseFilter',
    minRatio = cms.double(0.70),
    maxRatio = cms.double(0.96),
    minHPDHits = cms.int32(17),
    minRBXHits = cms.int32(999),
    minHPDNoOtherHits = cms.int32(10),
    minZeros = cms.int32(10),
    minHighEHitTime = cms.double(-9999.0),
    maxHighEHitTime = cms.double(9999.0),
    maxRBXEMF = cms.double(-9999.0)
    )
