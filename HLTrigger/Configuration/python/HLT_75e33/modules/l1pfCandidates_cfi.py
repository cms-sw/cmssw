import FWCore.ParameterSet.Config as cms

l1pfCandidates = cms.EDProducer("L1TPFCandMultiMerger",
    labelsToMerge = cms.vstring(
        'Calo',
        'TK',
        'TKVtx',
        'PF',
        'Puppi'
    ),
    pfProducers = cms.VInputTag(cms.InputTag("l1pfProducerBarrel"), cms.InputTag("l1pfProducerHGCal"), cms.InputTag("l1pfProducerHGCalNoTK"), cms.InputTag("l1pfProducerHF"))
)
