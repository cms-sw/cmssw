import FWCore.ParameterSet.Config as cms

# producer of rechits starting from tb simulation
ecalTBSimWeightUncalibRecHit = cms.EDProducer("EcalTBWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag('simEcalUnsuppressedDigis',''),
    EEdigiCollection = cms.InputTag('',''),                                              
    tdcRecInfoCollection = cms.InputTag('ecalTBSimTDCReconstructor','EcalTBTDCRecInfo'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    nbTimeBin = cms.int32(25),
)


