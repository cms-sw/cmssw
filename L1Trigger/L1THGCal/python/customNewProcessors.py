import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import layer1truncation_proc
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import stage1truncation_proc
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import truncation_params

def custom_layer1_truncation(process):
    parameters = layer1truncation_proc.clone()
    process.hgcalBackEndLayer1Producer.ProcessorParameters = parameters
    process.hgcalBackEndLayer2Producer.InputCluster = cms.InputTag('hgcalBackEndLayer1Producer:HGCalBackendLayer1Processor')
    process.hgcalTowerProducer.InputTriggerCells = cms.InputTag('hgcalBackEndLayer1Producer:HGCalBackendLayer1Processor')
    return process

def custom_stage1_truncation(process):
    parameters = stage1truncation_proc.clone()
    process.hgcalBackEndLayer1Producer.ProcessorParameters = parameters
    process.hgcalBackEndLayer2Producer.InputCluster = cms.InputTag('hgcalBackEndStage1Producer:HGCalBackendStage1Processor')
    process.hgcalTowerProducer.InputTriggerCells = cms.InputTag('hgcalBackEndStage1Producer:HGCalBackendStage1Processor')
    return process

def custom_clustering_standalone(process):
    process.hgcalBackEndLayer2Producer.ProcessorParameters.ProcessorName = cms.string('HGCalBackendLayer2Processor3DClusteringSA')
    process.hgcalBackEndLayer2Producer.ProcessorParameters.DistributionParameters = truncation_params
    return process

def custom_tower_standalone(process):
    process.hgcalTowerProducer.ProcessorParameters.ProcessorName = cms.string('HGCalTowerProcessorSA')
    return process
