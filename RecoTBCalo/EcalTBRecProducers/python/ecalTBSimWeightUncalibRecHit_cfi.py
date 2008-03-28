import FWCore.ParameterSet.Config as cms

# producer of rechits starting from tb simulation
ecalTBSimWeightUncalibRecHit = cms.EDProducer("EcalTBWeightUncalibRecHitProducer",
    EBdigiCollection = cms.string(''),
    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    digiProducer = cms.string('ecalUnsuppressedDigis'),
    nbTimeBin = cms.int32(25),
    tdcRecInfoProducer = cms.string('ecalTBSimTDCReconstructor')
)


