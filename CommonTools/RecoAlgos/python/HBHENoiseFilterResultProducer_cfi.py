import FWCore.ParameterSet.Config as cms

HBHENoiseFilterResultProducer = cms.EDProducer(
    'HBHENoiseFilterResultProducer',
    noiselabel = cms.InputTag('hcalnoise'),
    minHPDHits = cms.int32(17),
    minHPDNoOtherHits = cms.int32(10),
    minZeros = cms.int32(9999),
    IgnoreTS4TS5ifJetInLowBVRegion = cms.bool(True),
    defaultDecision = cms.string("HBHENoiseFilterResultRun1"),
    minNumIsolatedNoiseChannels = cms.int32(10),
    minIsolatedNoiseSumE = cms.double(50.0),
    minIsolatedNoiseSumEt = cms.double(25.0)
)

from Configuration.StandardSequences.Eras import eras
eras.run2_common.toModify(HBHENoiseFilterResultProducer, IgnoreTS4TS5ifJetInLowBVRegion=False)
eras.run2_25ns_specific.toModify(HBHENoiseFilterResultProducer, defaultDecision="HBHENoiseFilterResultRun2Loose")
