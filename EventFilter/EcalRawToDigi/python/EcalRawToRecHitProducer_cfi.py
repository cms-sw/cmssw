import FWCore.ParameterSet.Config as cms

EcalRawToRecHitProducer = cms.EDProducer("EcalRawToRecHitProducer",
    lazyGetterTag = cms.InputTag("EcalRawToRecHitFacility"),
    sourceTag = cms.InputTag("EcalRawToRecHitRoI"),
    splitOutput = cms.bool(True),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    EErechitCollection = cms.string('EcalRecHitsEE'),
    rechitCollection = cms.string('')
)


