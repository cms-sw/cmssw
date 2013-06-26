import FWCore.ParameterSet.Config as cms

ecal2007TBWeightUncalibRecHit = cms.EDProducer("EcalTBWeightUncalibRecHitProducer",
    use2004OffsetConvention = cms.untracked.bool(False),
    EBdigiCollection = cms.InputTag("",""),
    EEdigiCollection = cms.InputTag("ecalTBunpack","eeDigis"),
    tdcRecInfoCollection = cms.InputTag("ecal2007H4TBTDCReconstructor","EcalTBTDCRecInfo"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),                                               
    nbTimeBin = cms.int32(25),
)


