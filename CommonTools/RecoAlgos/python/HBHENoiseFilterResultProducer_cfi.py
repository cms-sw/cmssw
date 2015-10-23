import FWCore.ParameterSet.Config as cms

HBHENoiseFilterResultProducer = cms.EDProducer(
    'HBHENoiseFilterResultProducer',
    noiselabel = cms.InputTag('hcalnoise'),
    minHPDHits = cms.int32(17),
    minHPDNoOtherHits = cms.int32(10),
    minZeros = cms.int32(9999),
    IgnoreTS4TS5ifJetInLowBVRegion = cms.bool(True),
    defaultDecision = cms.string("HBHENoiseFilterResultRun2Loose"),
    minNumIsolatedNoiseChannels = cms.int32(10),
    minIsolatedNoiseSumE = cms.double(50.0),
    minIsolatedNoiseSumEt = cms.double(25.0),
    useBunchSpacingProducer = cms.bool(True)
)

from Configuration.StandardSequences.Eras import eras
eras.run2_common.toModify(HBHENoiseFilterResultProducer, IgnoreTS4TS5ifJetInLowBVRegion=False)
