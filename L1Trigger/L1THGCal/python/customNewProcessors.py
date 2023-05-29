import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.l1tHGCalBackEndLayer1Producer_cfi import layer1truncation_proc
from L1Trigger.L1THGCal.l1tHGCalBackEndLayer1Producer_cfi import stage1truncation_proc
from L1Trigger.L1THGCal.l1tHGCalBackEndLayer1Producer_cfi import truncation_params

def custom_layer1_truncation(process):
    parameters = layer1truncation_proc.clone()
    process.l1tHGCalBackEndLayer1Producer.ProcessorParameters = parameters
    process.l1tHGCalBackEndLayer2Producer.InputCluster = cms.InputTag('l1tHGCalBackEndLayer1Producer:HGCalBackendLayer1Processor')
    process.l1tHGCalTowerProducer.InputTriggerCells = cms.InputTag('l1tHGCalBackEndLayer1Producer:HGCalBackendLayer1Processor')
    return process

def custom_stage1_truncation(process):
    parameters = stage1truncation_proc.clone()
    process.l1tHGCalBackEndLayer1Producer.ProcessorParameters = parameters
    process.l1tHGcalBackEndLayer2Producer.InputCluster = cms.InputTag('l1tHGCalBackEndStage1Producer:HGCalBackendStage1Processor')
    process.l1tHGCalTowerProducer.InputTriggerCells = cms.InputTag('l1tHGCalBackEndStage1Producer:HGCalBackendStage1Processor')
    return process

def custom_clustering_standalone(process):
    process.l1tHGCalBackEndLayer2Producer.ProcessorParameters.ProcessorName = cms.string('HGCalBackendLayer2Processor3DClusteringSA')
    process.l1tHGCalBackEndLayer2Producer.ProcessorParameters.DistributionParameters = truncation_params
    return process

def custom_tower_standalone(process):
    process.l1tHGCalTowerProducer.ProcessorParameters.ProcessorName = cms.string('HGCalTowerProcessorSA')
    return process
