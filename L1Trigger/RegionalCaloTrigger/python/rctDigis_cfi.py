import FWCore.ParameterSet.Config as cms

rctDigis = cms.EDProducer("L1RCTProducer",
    hcalDigis = cms.VInputTag(cms.InputTag("hcalTriggerPrimitiveDigis")),
    useEcal = cms.bool(True),
    useHcal = cms.bool(True),
    ecalDigis = cms.VInputTag(cms.InputTag("ecalTriggerPrimitiveDigis")),
    BunchCrossings = cms.vint32(0),
    getFedsFromOmds = cms.bool(False),
#    getFedsFromOmds = cms.bool(True), # ONLY for online DQM!
    queryDelayInLS = cms.uint32(10),
    queryIntervalInLS = cms.uint32(100),
    conditionsLabel = cms.string("")
)



