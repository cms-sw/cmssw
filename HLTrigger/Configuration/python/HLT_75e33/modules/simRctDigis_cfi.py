import FWCore.ParameterSet.Config as cms

simRctDigis = cms.EDProducer("L1RCTProducer",
    BunchCrossings = cms.vint32(0),
    conditionsLabel = cms.string(''),
    ecalDigis = cms.VInputTag("simEcalTriggerPrimitiveDigis"),
    getFedsFromOmds = cms.bool(False),
    hcalDigis = cms.VInputTag("simHcalTriggerPrimitiveDigis"),
    queryDelayInLS = cms.uint32(10),
    queryIntervalInLS = cms.uint32(100),
    useEcal = cms.bool(True),
    useHcal = cms.bool(True)
)
