import FWCore.ParameterSet.Config as cms

ecal2007TBWeightUncalibRecHit = cms.EDProducer("EcalTBWeightUncalibRecHitProducer",
    use2004OffsetConvention = cms.untracked.bool(False),
    EBdigiCollection = cms.string('ebDigis'),
    EEigiCollection = cms.string('eeDigis'),
    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),                                               
    digiProducer = cms.string('ecalTBunpack'),
    nbTimeBin = cms.int32(25),
    tdcRecInfoProducer = cms.string('ecal2007H4TBTDCReconstructor')
)


