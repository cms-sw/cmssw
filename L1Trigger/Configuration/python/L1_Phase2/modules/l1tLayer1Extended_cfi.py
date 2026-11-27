import FWCore.ParameterSet.Config as cms

l1tLayer1Extended = cms.EDProducer("L1TPFCandMultiMerger",
    pfProducers = cms.VInputTag("l1tLayer1BarrelExtended", "l1tLayer1HGCalExtended", "l1tLayer1HGCalNoTK", "l1tLayer1HF")
)
