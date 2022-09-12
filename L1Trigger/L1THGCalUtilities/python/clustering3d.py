import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.l1tHGCalBackEndLayer2Producer_cfi import distance_C3d_params, \
                                                              dbscan_C3d_params, \
                                                              histoMax_C3d_clustering_params, \
                                                              histoMax_C3d_seeding_params, \
                                                              histoMaxVariableDR_C3d_params, \
                                                              histoMaxXYVariableDR_C3d_params, \
                                                              histoSecondaryMax_C3d_params, \
                                                              histoInterpolatedMax_C3d_params, \
                                                              histoThreshold_C3d_params, \
                                                              neighbour_weights_1stOrder, \
                                                              neighbour_weights_2ndOrder

from L1Trigger.L1THGCal.customClustering import set_histomax_clustering_params
from L1Trigger.L1THGCal.customHistoSeeding import set_histomax_seeding_params


def create_distance(process, inputs,
                    distance=distance_C3d_params.dR_multicluster
                    ):
    producer = process.l1tHGCalBackEndLayer2Producer.clone(
            InputCluster = cms.InputTag(inputs)
            )
    producer.ProcessorParameters.C3d_parameters = distance_C3d_params.clone(
            dR_multicluster = distance
            )
    return producer


def create_dbscan(process, inputs,
                  distance=dbscan_C3d_params.dist_dbscan_multicluster,
                  min_points=dbscan_C3d_params.minN_dbscan_multicluster
                  ):
    producer = process.l1tHGCalBackEndLayer2Producer.clone(
            InputCluster = cms.InputTag(inputs)
            )
    producer.ProcessorParameters.C3d_parameters = dbscan_C3d_params.clone(
            dist_dbscan_multicluster = distance,
            minN_dbscan_multicluster = min_points
            )
    return producer


class CreateHistoMax(object):
    def __init__(self,
            distance=histoMax_C3d_clustering_params.dR_multicluster,
            nBins_X1=histoMax_C3d_seeding_params.nBins_X1_histo_multicluster,
            nBins_X2=histoMax_C3d_seeding_params.nBins_X2_histo_multicluster,
            binSumsHisto=histoMax_C3d_seeding_params.binSumsHisto,
            seed_threshold=histoMax_C3d_seeding_params.threshold_histo_multicluster,
            shape_threshold=histoMax_C3d_clustering_params.shape_threshold,
            shape_distance=histoMax_C3d_clustering_params.shape_distance,
            ):
        self.clustering_parameters = histoMax_C3d_clustering_params.clone()
        self.seeding_parameters = histoMax_C3d_seeding_params.clone()
        set_histomax_seeding_params(self.seeding_parameters, nBins_X1, nBins_X2, binSumsHisto, seed_threshold)
        set_histomax_clustering_params(self.clustering_parameters, distance, shape_threshold, shape_distance)

    def __call__(self, process, inputs):
        producer = process.l1tHGCalBackEndLayer2Producer.clone(
                InputCluster = cms.InputTag(inputs)
                )
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters = self.clustering_parameters
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = self.seeding_parameters
        return producer


class CreateHistoMaxVariableDr(object):
    def __init__(self,
            distances=histoMaxVariableDR_C3d_params.dR_multicluster_byLayer_coefficientA,
            nBins_X1=histoMax_C3d_seeding_params.nBins_X1_histo_multicluster,
            nBins_X2=histoMax_C3d_seeding_params.nBins_X2_histo_multicluster,
            binSumsHisto=histoMax_C3d_seeding_params.binSumsHisto,
            seed_threshold=histoMax_C3d_seeding_params.threshold_histo_multicluster,
            shape_threshold=histoMaxVariableDR_C3d_params.shape_threshold,
            shape_distance=histoMaxVariableDR_C3d_params.shape_distance,
            ):
        self.clustering_parameters= histoMax_C3d_clustering_params.clone(
                dR_multicluster_byLayer_coefficientA = distances
                )
        self.seeding_parameters = histoMax_C3d_seeding_params.clone()
        set_histomax_seeding_params(self.seeding_parameters, nBins_X1, nBins_X2, binSumsHisto, seed_threshold)
        set_histomax_clustering_params(self.clustering_parameters, 0, shape_threshold, shape_distance)

    def __call__(self, process, inputs):
        producer = process.l1tHGCalBackEndLayer2Producer.clone(
                InputCluster = cms.InputTag(inputs)
                )
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters = self.clustering_parameters
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = self.seeding_parameters
        return producer


class CreateHistoMaxXYVariableDr(object):
    def __init__(self,
            distances=histoMaxVariableDR_C3d_params.dR_multicluster_byLayer_coefficientA,
            nBins_X1=histoMaxXYVariableDR_C3d_params.nBins_X1_histo_multicluster,
            nBins_X2=histoMaxXYVariableDR_C3d_params.nBins_X2_histo_multicluster,
            seed_threshold=histoMaxXYVariableDR_C3d_params.threshold_histo_multicluster,
            shape_threshold=histoMaxVariableDR_C3d_params.shape_threshold,
            shape_distance=histoMaxVariableDR_C3d_params.shape_distance,
            ):
        self.clustering_parameters = histoMax_C3d_clustering_params.clone(
                dR_multicluster_byLayer_coefficientA = distances
                )
        self.seeding_parameters = histoMaxXYVariableDR_C3d_params.clone()
        set_histomax_seeding_params(self.seeding_parameters, nBins_X1, nBins_X2, histoMaxXYVariableDR_C3d_params.binSumsHisto, seed_threshold)
        set_histomax_clustering_params(self.clustering_parameters, 0, shape_threshold, shape_distance)

    def __call__(self, process, inputs):
        producer = process.l1tHGCalBackEndLayer2Producer.clone(
                InputCluster = cms.InputTag(inputs)
                )
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters = self.clustering_parameters
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = self.seeding_parameters
        return producer


class CreateHistoInterpolatedMax1stOrder(object):
    def __init__(self,
            distance=histoMax_C3d_clustering_params.dR_multicluster,
            nBins_X1=histoInterpolatedMax_C3d_params.nBins_X1_histo_multicluster,
            nBins_X2=histoInterpolatedMax_C3d_params.nBins_X2_histo_multicluster,
            binSumsHisto=histoInterpolatedMax_C3d_params.binSumsHisto,
            seed_threshold=histoInterpolatedMax_C3d_params.threshold_histo_multicluster,
            shape_threshold=histoMax_C3d_clustering_params.shape_threshold,
            shape_distance=histoMax_C3d_clustering_params.shape_distance,
            ):
        self.seeding_parameters = histoInterpolatedMax_C3d_params.clone(
                neighbour_weights = neighbour_weights_1stOrder
                )
        self.clustering_parameters = histoMax_C3d_clustering_params.clone()
        set_histomax_seeding_params(self.seeding_parameters, nBins_X1, nBins_X2, binSumsHisto, seed_threshold)
        set_histomax_clustering_params(self.clustering_parameters, distance, shape_threshold, shape_distance)

    def __call__(self, process, inputs):
        producer = process.l1tHGCalBackEndLayer2Producer.clone(
                InputCluster = cms.InputTag(inputs)
                )
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = self.seeding_parameters
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters = self.clustering_parameters
        return producer


class CreateHistoInterpolatedMax2ndOrder(object):
    def __init__(self,
            distance=histoMax_C3d_clustering_params.dR_multicluster,
            nBins_X1=histoInterpolatedMax_C3d_params.nBins_X1_histo_multicluster,
            nBins_X2=histoInterpolatedMax_C3d_params.nBins_X2_histo_multicluster,
            binSumsHisto=histoInterpolatedMax_C3d_params.binSumsHisto,
            seed_threshold=histoInterpolatedMax_C3d_params.threshold_histo_multicluster,
            shape_threshold=histoMax_C3d_clustering_params.shape_threshold,
            shape_distance=histoMax_C3d_clustering_params.shape_distance,
            ):
        self.seeding_parameters = histoInterpolatedMax_C3d_params.clone(
                neighbour_weights = neighbour_weights_2ndOrder
                )
        self.clustering_parameters = histoMax_C3d_clustering_params.clone()
        set_histomax_seeding_params(self.seeding_parameters, nBins_X1, nBins_X2, binSumsHisto, seed_threshold)
        set_histomax_clustering_params(self.clustering_parameters, distance, shape_threshold, shape_distance)

    def __call__(self, process, inputs):
        producer = process.l1tHGCalBackEndLayer2Producer.clone(
                InputCluster = cms.InputTag(inputs)
                )
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = self.seeding_parameters
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters = self.clustering_parameters
        return producer


class CreateHistoThreshold(object):
    def __init__(self,
            seed_threshold=histoThreshold_C3d_params.threshold_histo_multicluster,
            distance=histoMax_C3d_clustering_params.dR_multicluster,
            nBins_X1=histoThreshold_C3d_params.nBins_X1_histo_multicluster,
            nBins_X2=histoThreshold_C3d_params.nBins_X2_histo_multicluster,
            binSumsHisto=histoThreshold_C3d_params.binSumsHisto,
            shape_threshold=histoMax_C3d_clustering_params.shape_threshold,
            shape_distance=histoMax_C3d_clustering_params.shape_distance,
            ):
        self.seeding_parameters = histoThreshold_C3d_params.clone()
        self.clustering_parameters = histoMax_C3d_clustering_params.clone()
        set_histomax_seeding_params(self.seeding_parameters, nBins_X1, nBins_X2, binSumsHisto, seed_threshold)
        set_histomax_clustering_params(self.clustering_parameters, distance, shape_threshold, shape_distance)

    def __call__(self, process, inputs):
        producer = process.l1tHGCalBackEndLayer2Producer.clone(
                InputCluster = cms.InputTag(inputs)
                )
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = self.seeding_parameters
        producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters = self.clustering_parameters
        return producer
