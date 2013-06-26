import FWCore.ParameterSet.Config as cms

ecal2006TBH2WeightUncalibRecHit = cms.EDProducer("EcalTBWeightUncalibRecHitProducer",
    use2004OffsetConvention = cms.untracked.bool(False),
    EBdigiCollection = cms.InputTag('ecalTBunpack',''),
    EEdigiCollection = cms.InputTag('ecalTBunpack',''),                                                 
    tdcRecInfoCollection = cms.InputTag('ecal2006TBH2TDCReconstructor','EcalTBTDCRecInfo'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    digiProducer = cms.string('ecalTBunpack'),
    nbTimeBin = cms.int32(25),
)


