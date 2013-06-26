import FWCore.ParameterSet.Config as cms

HBHENoiseFilterResultProducer = cms.EDProducer(
    'HBHENoiseFilterResultProducer',
    noiselabel = cms.InputTag('hcalnoise'),
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
    useTS4TS5 = cms.bool(True),
    IgnoreTS4TS5ifJetInLowBVRegion=cms.bool(True),
    jetlabel = cms.InputTag('ak5PFJets'),
    maxjetindex = cms.int32(0), # maximum jet index that will be checked for 'IgnoreTS4TS5ifJetInLowBVRegion'
    maxNHF = cms.double(0.9) # maximum allowed jet->neutralHadronEnergyFraction() for a jet in low BV region to be considered 'good' (and thus skip the noise check)
    )
