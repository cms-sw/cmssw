import FWCore.ParameterSet.Config as cms
import L1Trigger.L1THGCal.hgcalTowerMapProducer_cfi as hgcalTowerMapProducer_cfi

tower = cms.PSet( ProcessorName  = cms.string('HGCalTowerProcessor'),
      towermap_parameters = hgcalTowerMapProducer_cfi.towerMap2D_parValues.clone()
                  )

hgcalTowerProducer = cms.EDProducer(
    "HGCalTowerProducer",
    InputTowerMaps = cms.InputTag('hgcalTowerMapProducer:HGCalTowerMapProcessor'), 
    InputTriggerCells = cms.InputTag('hgcalBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering'),
    ProcessorParameters = tower.clone(),
    )


hgcalTowerProducerHFNose = hgcalTowerProducer.clone(
    InputTowerMaps = cms.InputTag('hgcalTowerMapProducerHFNose:HGCalTowerMapProcessor'),
    InputTriggerCells = cms.InputTag('hgcalBackEndLayer1ProducerHFNose:HGCalBackendLayer1Processor2DClustering'),
)

