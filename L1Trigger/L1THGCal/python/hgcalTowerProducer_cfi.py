import FWCore.ParameterSet.Config as cms

tower = cms.PSet( ProcessorName  = cms.string('HGCalTowerProcessor')
                  )

hgcalTowerProducer = cms.EDProducer(
    "HGCalTowerProducer",
    InputTriggerCells = cms.InputTag('hgcalTowerMapProducer:HGCalTowerMapProcessor'), 
    ProcessorParameters = tower.clone()
    )
