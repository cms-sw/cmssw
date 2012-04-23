import FWCore.ParameterSet.Config as cms

HBHENoiseFilterResultProducer = cms.EDProducer(
    'HBHENoiseFilterResultProducer',
    label = cms.InputTag('hcalnoise'),
    minRatio = cms.double(-999),
    maxRatio = cms.double(999),
    minHPDHits = cms.int32(17),
    minRBXHits = cms.int32(999),
    minHPDNoOtherHits = cms.int32(10),
    minZeros = cms.int32(10),
    minHighEHitTime = cms.double(-9999.0),
    maxHighEHitTime = cms.double(9999.0),
    maxRBXEMF = cms.double(-999.0),
    minNumIsolatedNoiseChannels = cms.int32(10),
    minIsolatedNoiseSumE = cms.double(50.0),
    minIsolatedNoiseSumEt = cms.double(25.0),
    useTS4TS5 = cms.bool(True)
    )
