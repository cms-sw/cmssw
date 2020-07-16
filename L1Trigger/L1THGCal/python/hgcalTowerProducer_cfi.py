import FWCore.ParameterSet.Config as cms

tower = cms.PSet( ProcessorName  = cms.string('HGCalTowerProcessor')
                  )

hgcalTowerProducer = cms.EDProducer(
    "HGCalTowerProducer",
    InputTowerMaps = cms.InputTag('hgcalTowerMapProducer:HGCalTowerMapProcessor'), 
    ProcessorParameters = tower.clone()
    )


hgcalTowerProducerHFNose = hgcalTowerProducer.clone(
    InputTowerMaps = cms.InputTag('hgcalTowerMapProducerHFNose:HGCalTowerMapProcessor')
)

