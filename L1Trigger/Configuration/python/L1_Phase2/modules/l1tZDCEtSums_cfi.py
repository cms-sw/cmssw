import FWCore.ParameterSet.Config as cms

l1tZDCEtSums = cms.EDProducer("L1TZDCEtSumsProducer",
    bxFirst = cms.int32(-2),
    bxLast = cms.int32(3),
    hcalTPDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    mightGet = cms.optional.untracked.vstring
)
