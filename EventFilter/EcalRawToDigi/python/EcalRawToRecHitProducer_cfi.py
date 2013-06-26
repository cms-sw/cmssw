import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.EcalRecAlgos.ecalCleaningAlgo import cleaningAlgoConfig
 
EcalRawToRecHitProducer = cms.EDProducer("EcalRawToRecHitProducer",
    cleaningConfig=cleaningAlgoConfig,
    lazyGetterTag = cms.InputTag("EcalRawToRecHitFacility"),
    sourceTag = cms.InputTag("EcalRawToRecHitRoI"),
    splitOutput = cms.bool(True),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    EErechitCollection = cms.string('EcalRecHitsEE'),
    rechitCollection = cms.string('')
)


