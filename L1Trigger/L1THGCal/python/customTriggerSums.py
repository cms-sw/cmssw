import FWCore.ParameterSet.Config as cms

def custom_partial_trigger_sums(process):
    process.tower.includeTrigCells = cms.bool(True)
    process.l1tHGCalTowerProducer.ProcessorParameters.includeTrigCells = cms.bool(True)
    process.l1tHGCalTowerProducerHFNose.ProcessorParameters.includeTrigCells = cms.bool(True)
    process.threshold_conc_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.best_conc_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.supertc_conc_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.custom_conc_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.coarsetc_onebitfraction_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.coarsetc_equalshare_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.autoEncoder_conc_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.l1tHGCalConcentratorProducer.ProcessorParameters.allTrigCellsInTrigSums = cms.bool(False)
    process.l1tHGCalConcentratorProducerHFNose.ProcessorParameters.allTrigCellsInTrigSums = cms.bool(False)
    return process

def custom_full_trigger_sums(process):
    process.tower.includeTrigCells = cms.bool(False)
    process.l1tHGCalTowerProducer.ProcessorParameters.includeTrigCells = cms.bool(False)
    process.l1tHGCalTowerProducerHFNose.ProcessorParameters.includeTrigCells = cms.bool(False)
    process.threshold_conc_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.best_conc_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.supertc_conc_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.custom_conc_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.coarsetc_onebitfraction_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.coarsetc_equalshare_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.autoEncoder_conc_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.l1tHGCalConcentratorProducer.ProcessorParameters.allTrigCellsInTrigSums = cms.bool(True)
    process.l1tHGCalConcentratorProducerHFNose.ProcessorParameters.allTrigCellsInTrigSums = cms.bool(True)
    return process
