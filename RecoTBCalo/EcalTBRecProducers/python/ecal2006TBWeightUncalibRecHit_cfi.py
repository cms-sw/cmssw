import FWCore.ParameterSet.Config as cms

ecal2006TBWeightUncalibRecHit = cms.EDProducer("EcalTBWeightUncalibRecHitProducer",
    use2004OffsetConvention = cms.untracked.bool(False),
    EBdigiCollection = cms.InputTag('ecalTBunpack',''),
    EEdigiCollection = cms.InputTag('',''),                                               
    tdcRecInfoCollection = cms.InputTag('ecal2006TBTDCReconstructor','EcalTBTDCRecInfo'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    nbTimeBin = cms.int32(25),
)


