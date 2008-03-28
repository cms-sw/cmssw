import FWCore.ParameterSet.Config as cms

ecal2006TBWeightUncalibRecHit = cms.EDProducer("EcalTBWeightUncalibRecHitProducer",
    use2004OffsetConvention = cms.untracked.bool(False),
    EBdigiCollection = cms.string(''),
    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    digiProducer = cms.string('ecalTBunpack'),
    nbTimeBin = cms.int32(25),
    tdcRecInfoProducer = cms.string('ecal2006TBTDCReconstructor')
)


