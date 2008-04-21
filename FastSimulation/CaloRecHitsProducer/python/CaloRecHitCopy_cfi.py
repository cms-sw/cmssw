import FWCore.ParameterSet.Config as cms

caloRecHitCopy = cms.EDFilter("CaloRecHitCopy",
    InputRecHitCollectionTypes = cms.vuint32(2, 3),
    OutputRecHitCollections = cms.vstring('EcalRecHitsEB', 
        'EcalRecHitsEE'),
    InputRecHitCollections = cms.VInputTag(cms.InputTag("caloRecHits","EcalRecHitsEB"), cms.InputTag("caloRecHits","EcalRecHitsEE"))
)


