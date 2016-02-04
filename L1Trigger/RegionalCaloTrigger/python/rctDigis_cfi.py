import FWCore.ParameterSet.Config as cms

rctDigis = cms.EDProducer("L1RCTProducer",
    hcalDigis = cms.VInputTag(cms.InputTag("hcalTriggerPrimitiveDigis")),
    useDebugTpgScales = cms.bool(False),
    useEcal = cms.bool(True),
    useHcal = cms.bool(True),
    ecalDigis = cms.VInputTag(cms.InputTag("ecalTriggerPrimitiveDigis")),
    BunchCrossings = cms.vint32(0)                      
)



