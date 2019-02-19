import FWCore.ParameterSet.Config as cms


def create_distance(process, inputs,
        distance=6.,# cm
        seed_threshold=5.,# MipT
        cluster_threshold=2.# MipT
        ):
    producer = process.hgcalBackEndLayer1Producer.clone() 
    producer.ProcessorParameters.C2d_parameters.seeding_threshold_silicon = cms.double(seed_threshold) 
    producer.ProcessorParameters.C2d_parameters.seeding_threshold_scintillator = cms.double(seed_threshold) 
    producer.ProcessorParameters.C2d_parameters.clustering_threshold_silicon = cms.double(cluster_threshold) 
    producer.ProcessorParameters.C2d_parameters.clustering_threshold_scintillator = cms.double(cluster_threshold) 
    producer.ProcessorParameters.C2d_parameters.dR_cluster = cms.double(distance) 
    producer.ProcessorParameters.C2d_parameters.clusterType = cms.string('dRC2d') 
    producer.InputTriggerCells = cms.InputTag('{}:HGCalConcentratorProcessorSelection'.format(inputs))
    return producer

def create_topological(process, inputs,
        seed_threshold=5.,# MipT
        cluster_threshold=2.# MipT
        ):
    producer = process.hgcalBackEndLayer1Producer.clone() 
    producer.ProcessorParameters.C2d_parameters.seeding_threshold_silicon = cms.double(seed_threshold) # MipT
    producer.ProcessorParameters.C2d_parameters.seeding_threshold_scintillator = cms.double(seed_threshold) # MipT
    producer.ProcessorParameters.C2d_parameters.clustering_threshold_silicon = cms.double(cluster_threshold) # MipT
    producer.ProcessorParameters.C2d_parameters.clustering_threshold_scintillator = cms.double(cluster_threshold) # MipT
    producer.ProcessorParameters.C2d_parameters.clusterType = cms.string('NNC2d') 
    producer.InputTriggerCells = cms.InputTag('{}:HGCalConcentratorProcessorSelection'.format(inputs))
    return producer

def create_constrainedtopological(process, inputs,
        distance=6.,# cm
        seed_threshold=5.,# MipT
        cluster_threshold=2.# MipT
        ):
    producer = process.hgcalBackEndLayer1Producer.clone() 
    producer.ProcessorParameters.C2d_parameters.seeding_threshold_silicon = cms.double(seed_threshold) # MipT
    producer.ProcessorParameters.C2d_parameters.seeding_threshold_scintillator = cms.double(seed_threshold) # MipT
    producer.ProcessorParameters.C2d_parameters.clustering_threshold_silicon = cms.double(cluster_threshold) # MipT
    producer.ProcessorParameters.C2d_parameters.clustering_threshold_scintillator = cms.double(cluster_threshold) # MipT
    producer.ProcessorParameters.C2d_parameters.dR_cluster = cms.double(distance) # cm
    producer.ProcessorParameters.C2d_parameters.clusterType = cms.string('dRNNC2d') 
    producer.InputTriggerCells = cms.InputTag('{}:HGCalConcentratorProcessorSelection'.format(inputs))
    return producer

def create_dummy(process, inputs):
    producer = process.hgcalBackEndLayer1Producer.clone() 
    producer.ProcessorParameters.C2d_parameters.clusterType = cms.string('dummyC2d')
    producer.InputTriggerCells = cms.InputTag('{}:HGCalConcentratorProcessorSelection'.format(inputs))
    return producer
