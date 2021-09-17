import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import dummy_C2d_params, \
                                                              distance_C2d_params, \
                                                              topological_C2d_params, \
                                                              constrTopological_C2d_params
from L1Trigger.L1THGCal.customClustering import set_threshold_params


def create_distance(process, inputs,
                    distance=distance_C2d_params.dR_cluster,  # cm
                    seed_threshold=distance_C2d_params.seeding_threshold_silicon,  # MipT
                    cluster_threshold=distance_C2d_params.clustering_threshold_silicon  # MipT
                    ):
    producer = process.hgcalBackEndLayer1Producer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalConcentratorProcessorSelection'.format(inputs))
            )
    producer.ProcessorParameters.C2d_parameters = distance_C2d_params.clone(
            dR_cluster = distance
            )
    set_threshold_params(producer.ProcessorParameters.C2d_parameters, seed_threshold, cluster_threshold)
    return producer


def create_topological(process, inputs,
                       seed_threshold=topological_C2d_params.seeding_threshold_silicon,  # MipT
                       cluster_threshold=topological_C2d_params.clustering_threshold_silicon  # MipT
                       ):
    producer = process.hgcalBackEndLayer1Producer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalConcentratorProcessorSelection'.format(inputs))
            )
    producer.ProcessorParameters.C2d_parameters = topological_C2d_params.clone()
    set_threshold_params(producer.ProcessorParameters.C2d_parameters, seed_threshold, cluster_threshold)
    return producer


def create_constrainedtopological(process, inputs,
                                  distance=constrTopological_C2d_params.dR_cluster,  # cm
                                  seed_threshold=constrTopological_C2d_params.seeding_threshold_silicon,  # MipT
                                  cluster_threshold=constrTopological_C2d_params.clustering_threshold_silicon  # MipT
                                  ):
    producer = process.hgcalBackEndLayer1Producer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalConcentratorProcessorSelection'.format(inputs))
            )
    producer.ProcessorParameters.C2d_parameters = constrTopological_C2d_params.clone(
            dR_cluster = distance
            )
    set_threshold_params(producer.ProcessorParameters.C2d_parameters, seed_threshold, cluster_threshold)
    return producer


def create_dummy(process, inputs):
    producer = process.hgcalBackEndLayer1Producer.clone(
            InputTriggerCells = cms.InputTag('{}:HGCalConcentratorProcessorSelection'.format(inputs))
            )
    producer.ProcessorParameters.C2d_parameters = dummy_C2d_params.clone()
    return producer

def create_truth_dummy(process, inputs):
    producer = process.hgcalBackEndLayer1Producer.clone(
            InputTriggerCells = cms.InputTag('{}'.format(inputs))
            )
    producer.ProcessorParameters.C2d_parameters = dummy_C2d_params.clone()
    return producer
