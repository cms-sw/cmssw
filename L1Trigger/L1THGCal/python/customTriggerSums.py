import FWCore.ParameterSet.Config as cms

def custom_partial_trigger_sums(process):
    process.tower.includeTrigCells = cms.bool(True)
    process.hgcalTowerProducer.ProcessorParameters.includeTrigCells = cms.bool(True)
    process.hgcalTowerProducerHFNose.ProcessorParameters.includeTrigCells = cms.bool(True)
    process.threshold_conc_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.best_conc_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.supertc_conc_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.custom_conc_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.coarsetc_onebitfraction_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.coarsetc_equalshare_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.autoEncoder_conc_proc.allTrigCellsInTrigSums = cms.bool(False)
    process.hgcalConcentratorProducer.ProcessorParameters.allTrigCellsInTrigSums = cms.bool(False)
    process.hgcalConcentratorProducerHFNose.ProcessorParameters.allTrigCellsInTrigSums = cms.bool(False)
    return process

def custom_full_trigger_sums(process):
    process.tower.includeTrigCells = cms.bool(False)
    process.hgcalTowerProducer.ProcessorParameters.includeTrigCells = cms.bool(False)
    process.hgcalTowerProducerHFNose.ProcessorParameters.includeTrigCells = cms.bool(False)
    process.threshold_conc_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.best_conc_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.supertc_conc_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.custom_conc_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.coarsetc_onebitfraction_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.coarsetc_equalshare_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.autoEncoder_conc_proc.allTrigCellsInTrigSums = cms.bool(True)
    process.hgcalConcentratorProducer.ProcessorParameters.allTrigCellsInTrigSums = cms.bool(True)
    process.hgcalConcentratorProducerHFNose.ProcessorParameters.allTrigCellsInTrigSums = cms.bool(True)
    return process
