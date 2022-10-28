import FWCore.ParameterSet.Config as cms
import L1Trigger.L1THGCal.l1tHGCalTowerMapProducer_cfi as hgcalTowerMapProducer_cfi

tower = cms.PSet( ProcessorName  = cms.string('HGCalTowerProcessor'),
      includeTrigCells = cms.bool(False),
      towermap_parameters = hgcalTowerMapProducer_cfi.towerMap2D_parValues.clone()
                  )

l1tHGCalTowerProducer = cms.EDProducer(
    "HGCalTowerProducer",
    InputTowerMaps = cms.InputTag('l1tHGCalTowerMapProducer:HGCalTowerMapProcessor'), 
    InputTriggerCells = cms.InputTag('l1tHGCalBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering'),
    ProcessorParameters = tower.clone(),
    )

towerHFNose = tower.clone(
    towermap_parameters = hgcalTowerMapProducer_cfi.towerMap2DHFNose_parValues.clone()
)

l1tHGCalTowerProducerHFNose = l1tHGCalTowerProducer.clone(
    InputTowerMaps = cms.InputTag('l1tHGCalTowerMapProducerHFNose:HGCalTowerMapProcessor'),
    InputTriggerCells = cms.InputTag('l1tHGCalBackEndLayer1ProducerHFNose:HGCalBackendLayer1Processor2DClustering'),
    ProcessorParameters = towerHFNose.clone(),
)

