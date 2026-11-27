import FWCore.ParameterSet.Config as cms

l1tLayer1 = cms.EDProducer("L1TPFCandMultiMerger",
    pfProducers = cms.VInputTag(cms.InputTag("l1tLayer1Barrel"), cms.InputTag("l1tLayer1HGCal"), cms.InputTag("l1tLayer1HGCalNoTK"), cms.InputTag("l1tLayer1HF"))
)
