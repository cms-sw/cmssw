import FWCore.ParameterSet.Config as cms

l1tPFCandidates = cms.EDProducer("L1TPFCandMultiMerger",
    labelsToMerge = cms.vstring(
        'Calo',
        'TK',
        'TKVtx',
        'PF',
        'Puppi'
    ),
    pfProducers = cms.VInputTag(cms.InputTag("l1tPFProducerBarrel"), cms.InputTag("l1tPFProducerHGCal"), cms.InputTag("l1tPFProducerHGCalNoTK"), cms.InputTag("l1tPFProducerHF"))
)
