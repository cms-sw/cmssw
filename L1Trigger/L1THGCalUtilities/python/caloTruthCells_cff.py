import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCalUtilities.caloTruthCellsProducer_cfi import l1tCaloTruthCellsProducer

L1TCaloTruthCells = cms.Sequence(l1tCaloTruthCellsProducer)

if l1tCaloTruthCellsProducer.makeCellsCollection:
    ## cluster and tower sequence

    from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import l1tHGCalConcentratorProducer
    from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import l1tHGCalBackEndLayer1Producer
    from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import l1tHGCalBackEndLayer2Producer
    from L1Trigger.L1THGCal.hgcalTowerMapProducer_cfi import l1tHGCalTowerMapProducer
    from L1Trigger.L1THGCal.hgcalTowerProducer_cfi import l1tHGCalTowerProducer
    
    l1tHGCalTruthConcentratorProducer = l1tHGCalConcentratorProducer.clone(
        InputTriggerCells = cms.InputTag('l1tCaloTruthCellsProducer')
    )
    
    l1tHGCalTruthBackEndLayer1Producer = l1tHGCalBackEndLayer1Producer.clone(
        InputTriggerCells = cms.InputTag('l1tHGCalTruthConcentratorProducer:HGCalConcentratorProcessorSelection')
    )
    
    l1tHGCalTruthBackEndLayer2Producer = l1tHGCalBackEndLayer2Producer.clone(
        InputCluster = cms.InputTag('l1tHGCalTruthBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering')
    )
    
    l1tHGCalTruthTowerMapProducer = l1tHGCalTowerMapProducer.clone(
        InputTriggerCells = cms.InputTag('l1tCaloTruthCellsProducer')
    )
    
    l1tHGCalTruthTowerProducer = l1tHGCalTowerProducer.clone(
        InputTowerMaps = cms.InputTag('l1tHGCalTruthTowerMapProducer:HGCalTowerMapProcessor')
    )
    
    L1TCaloTruthCells += cms.Sequence(
        l1tHGCalTruthConcentratorProducer *
        l1tHGCalTruthBackEndLayer1Producer *
        l1tHGCalTruthBackEndLayer2Producer *
        l1tHGCalTruthTowerMapProducer *
        l1tHGCalTruthTowerProducer
    )
