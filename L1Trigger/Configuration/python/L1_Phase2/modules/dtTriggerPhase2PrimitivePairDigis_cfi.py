import FWCore.ParameterSet.Config as cms

dtTriggerPhase2PrimitivePairDigis = cms.EDProducer("DTTrigPhase2PairsProd",
    debug = cms.untracked.bool(False),
    digiPhTag = cms.InputTag("dtTriggerPhase2PrimitiveDigis"),
    digiThTag = cms.InputTag("dtTriggerPhase2PrimitiveDigis"),
    scenario = cms.int32(0)
)
