import FWCore.ParameterSet.Config as cms

l1tLayer1EG = cms.EDProducer("L1TEGMultiMerger",
    tkEgs = cms.VPSet(cms.PSet(
        instance = cms.string('L1EgEE'),
        pfProducers = cms.VInputTag(cms.InputTag("l1tLayer1HGCal","L1Eg"), cms.InputTag("l1tLayer1HGCalNoTK","L1Eg"))
    )),
    tkElectrons = cms.VPSet(
        cms.PSet(
            instance = cms.string('L1TkEleEE'),
            pfProducers = cms.VInputTag(cms.InputTag("l1tLayer1HGCal","L1TkEle"))
        ),
        cms.PSet(
            instance = cms.string('L1TkEleEB'),
            pfProducers = cms.VInputTag(cms.InputTag("l1tLayer1Barrel","L1TkEle"))
        )
    ),
    tkEms = cms.VPSet(
        cms.PSet(
            instance = cms.string('L1TkEmEE'),
            pfProducers = cms.VInputTag(cms.InputTag("l1tLayer1HGCal","L1TkEm"), cms.InputTag("l1tLayer1HGCalNoTK","L1TkEm"))
        ),
        cms.PSet(
            instance = cms.string('L1TkEmEB'),
            pfProducers = cms.VInputTag(cms.InputTag("l1tLayer1Barrel","L1TkEm"))
        )
    )
)
