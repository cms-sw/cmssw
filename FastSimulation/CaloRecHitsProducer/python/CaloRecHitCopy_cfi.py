import FWCore.ParameterSet.Config as cms

caloRecHitCopy = cms.EDProducer("CaloRecHitCopy",
    InputRecHitCollectionTypes = cms.vuint32(2, 3),
    OutputRecHitCollections = cms.vstring('EcalRecHitsEB', 
        'EcalRecHitsEE'),
    InputRecHitCollections = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE"))
)


