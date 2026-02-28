
import FWCore.ParameterSet.Config as cms

dtTriggerPhase2PrimitivePairDigis = cms.EDProducer("DTTrigPhase2PairsProd",
                                               scenario = cms.int32(0), #0 for mc, 1 for data, 2 for slice test
                                               digiPhTag = cms.InputTag("dtTriggerPhase2PrimitiveDigis",""),
                                               digiThTag  = cms.InputTag("dtTriggerPhase2PrimitiveDigis",""),

                                               #debugging
                                               debug = cms.untracked.bool(False),
)
