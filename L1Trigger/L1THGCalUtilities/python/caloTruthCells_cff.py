import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCalUtilities.caloTruthCellsProducer_cfi import caloTruthCellsProducer

caloTruthCells = cms.Sequence(caloTruthCellsProducer)

if caloTruthCellsProducer.makeCellsCollection:
    ## cluster and tower sequence

    from L1Trigger.L1THGCal.hgcalConcentratorProducer_cfi import hgcalConcentratorProducer
    from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import hgcalBackEndLayer1Producer
    from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import hgcalBackEndLayer2Producer
    from L1Trigger.L1THGCal.hgcalTowerMapProducer_cfi import hgcalTowerMapProducer
    from L1Trigger.L1THGCal.hgcalTowerProducer_cfi import hgcalTowerProducer
    
    hgcalTruthConcentratorProducer = hgcalConcentratorProducer.clone(
        InputTriggerCells = cms.InputTag('caloTruthCellsProducer')
    )
    
    hgcalTruthBackEndLayer1Producer = hgcalBackEndLayer1Producer.clone(
        InputTriggerCells = cms.InputTag('hgcalTruthConcentratorProducer:HGCalConcentratorProcessorSelection')
    )
    
    hgcalTruthBackEndLayer2Producer = hgcalBackEndLayer2Producer.clone(
        InputCluster = cms.InputTag('hgcalTruthBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering')
    )
    
    hgcalTruthTowerMapProducer = hgcalTowerMapProducer.clone(
        InputTriggerCells = cms.InputTag('caloTruthCellsProducer')
    )
    
    hgcalTruthTowerProducer = hgcalTowerProducer.clone(
        InputTowerMaps = cms.InputTag('hgcalTruthTowerMapProducer:HGCalTowerMapProcessor')
    )
    
    caloTruthCells += cms.Sequence(
        hgcalTruthConcentratorProducer *
        hgcalTruthBackEndLayer1Producer *
        hgcalTruthBackEndLayer2Producer *
        hgcalTruthTowerMapProducer *
        hgcalTruthTowerProducer
    )
