import FWCore.ParameterSet.Config as cms

ecal2004TBWeightUncalibRecHit = cms.EDProducer("EcalTBWeightUncalibRecHitProducer",
    use2004OffsetConvention = cms.untracked.bool(True),
    EBdigiCollection = cms.InputTag('source',''),
    EEdigiCollection = cms.InputTag('',''),                                               
    tdcRecInfoCollection = cms.InputTag('ecal2004TBTDCReconstructor','EcalTBTDCRecInfo'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    nbTimeBin = cms.int32(25),
)


