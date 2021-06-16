import FWCore.ParameterSet.Config as cms

def custom_clustering_standalone(process):
    process.hgcalBackEndLayer2Producer.ProcessorParameters.ProcessorName = cms.string('HGCalBackendLayer2Processor3DClusteringSA')
    return process

def custom_tower_standalone(process):
    process.hgcalTowerProducer.ProcessorParameters.ProcessorName = cms.string('HGCalTowerProcessorSA')
    return process

