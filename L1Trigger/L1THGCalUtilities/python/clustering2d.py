import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.l1tHGCalBackEndLayer1Producer_cfi import dummy_C2d_params, \
                                                              distance_C2d_params, \
                                                              topological_C2d_params, \
                                                              constrTopological_C2d_params, \
                                                              layer1truncation_proc, \
                                                              truncation_params
from L1Trigger.L1THGCal.customClustering import set_threshold_params


def create_distance(process, inputs,
                    distance=distance_C2d_params.dR_cluster,  # cm
                    seed_threshold=distance_C2d_params.seeding_threshold_silicon,  # MipT
                    cluster_threshold=distance_C2d_params.clustering_threshold_silicon  # MipT
                    ):
    producer = process.l1tHGCalBackEndLayer1Producer.clone(
            InputTriggerCells = cms.InputTag(inputs)
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
    producer = process.l1tHGCalBackEndLayer1Producer.clone(
            InputTriggerCells = cms.InputTag(inputs)
            )
    producer.ProcessorParameters.C2d_parameters = topological_C2d_params.clone()
    set_threshold_params(producer.ProcessorParameters.C2d_parameters, seed_threshold, cluster_threshold)
    return producer


def create_constrainedtopological(process, inputs,
                                  distance=constrTopological_C2d_params.dR_cluster,  # cm
                                  seed_threshold=constrTopological_C2d_params.seeding_threshold_silicon,  # MipT
                                  cluster_threshold=constrTopological_C2d_params.clustering_threshold_silicon  # MipT
                                  ):
    producer = process.l1tHGCalBackEndLayer1Producer.clone(
            InputTriggerCells = cms.InputTag(inputs)
            )
    producer.ProcessorParameters.C2d_parameters = constrTopological_C2d_params.clone(
            dR_cluster = distance
            )
    set_threshold_params(producer.ProcessorParameters.C2d_parameters, seed_threshold, cluster_threshold)
    return producer



class CreateDummy(object):
    def __call__(self, process, inputs):
        producer = process.l1tHGCalBackEndLayer1Producer.clone(
                InputTriggerCells = cms.InputTag(inputs)
                )
        producer.ProcessorParameters.C2d_parameters = dummy_C2d_params.clone()
        return producer

class CreateTruthDummy(object):
    def __call__(self, process, inputs):
        producer = process.l1tHGCalBackEndLayer1Producer.clone(
                InputTriggerCells = cms.InputTag(inputs)
                )
        producer.ProcessorParameters.C2d_parameters = dummy_C2d_params.clone()
        return producer


class RozBinTruncation(object):
    def __init__(self,
            maxTcsPerBin=truncation_params.maxTcsPerBin):
        self.processor = layer1truncation_proc.clone(
                truncation_parameters=truncation_params.clone(
                    maxTcsPerBin=maxTcsPerBin
                    )
                )

    def __call__(self, process, inputs):
        producer = process.l1tHGCalBackEndLayer1Producer.clone(
                InputTriggerCells = cms.InputTag(inputs),
                ProcessorParameters = self.processor
                )
        return producer
