import FWCore.ParameterSet.Config as cms

rctDigis = cms.EDProducer("L1RCTProducer",
    hcalDigisLabel = cms.InputTag("hcalTriggerPrimitiveDigis"),
    useDebugTpgScales = cms.bool(False),
    useEcalCosmicTiming = cms.bool(False),
    postSamples = cms.uint32(0),
    preSamples = cms.uint32(0),
    useHcalCosmicTiming = cms.bool(False),
    useEcal = cms.bool(True),
    useHcal = cms.bool(True),
    ecalDigisLabel = cms.InputTag("ecalTriggerPrimitiveDigis"),
    UseMCAsInput = cms.bool(True),
    HFShift = cms.int32(-2),
    HBShift = cms.int32(1)
)



