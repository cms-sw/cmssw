import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import distance_C3d_params, \
                                                              dbscan_C3d_params, \
                                                              histoMax_C3d_params, \
                                                              histoMaxVariableDR_C3d_params, \
                                                              histoSecondaryMax_C3d_params, \
                                                              histoInterpolatedMax_C3d_params, \
                                                              histoThreshold_C3d_params, \
                                                              neighbour_weights_1stOrder, \
                                                              neighbour_weights_2ndOrder

from L1Trigger.L1THGCal.customClustering import set_histomax_params


def create_distance(process, inputs,
                    distance=distance_C3d_params.dR_multicluster
                    ):
    producer = process.hgcalBackEndLayer2Producer.clone(
            InputCluster = cms.InputTag('{}:HGCalBackendLayer1Processor2DClustering'.format(inputs))
            )
    producer.ProcessorParameters.C3d_parameters = distance_C3d_params.clone(
            dR_multicluster = distance
            )
    return producer


def create_dbscan(process, inputs,
                  distance=dbscan_C3d_params.dist_dbscan_multicluster,
                  min_points=dbscan_C3d_params.minN_dbscan_multicluster
                  ):
    producer = process.hgcalBackEndLayer2Producer.clone(
            InputCluster = cms.InputTag('{}:HGCalBackendLayer1Processor2DClustering'.format(inputs))
            )
    producer.ProcessorParameters.C3d_parameters = dbscan_C3d_params.clone(
            dist_dbscan_multicluster = distance,
            minN_dbscan_multicluster = min_points
            )
    return producer


def create_histoMax(process, inputs,
                    distance=histoMax_C3d_params.dR_multicluster,
                    nBins_R=histoMax_C3d_params.nBins_R_histo_multicluster,
                    nBins_Phi=histoMax_C3d_params.nBins_Phi_histo_multicluster,
                    binSumsHisto=histoMax_C3d_params.binSumsHisto,
                    seed_threshold=histoMax_C3d_params.threshold_histo_multicluster,
                    ):
    producer = process.hgcalBackEndLayer2Producer.clone(
            InputCluster = cms.InputTag('{}:HGCalBackendLayer1Processor2DClustering'.format(inputs))
            )
    producer.ProcessorParameters.C3d_parameters = histoMax_C3d_params.clone()
    set_histomax_params(producer.ProcessorParameters.C3d_parameters, distance, nBins_R, nBins_Phi, binSumsHisto, seed_threshold)
    return producer


def create_histoMax_variableDr(process, inputs,
                               distances=histoMaxVariableDR_C3d_params.dR_multicluster_byLayer_coefficientA,
                               nBins_R=histoMaxVariableDR_C3d_params.nBins_R_histo_multicluster,
                               nBins_Phi=histoMaxVariableDR_C3d_params.nBins_Phi_histo_multicluster,
                               binSumsHisto=histoMaxVariableDR_C3d_params.binSumsHisto,
                               seed_threshold=histoMaxVariableDR_C3d_params.threshold_histo_multicluster,
                               ):
    producer = process.hgcalBackEndLayer2Producer.clone(
            InputCluster = cms.InputTag('{}:HGCalBackendLayer1Processor2DClustering'.format(inputs))
            )
    producer.ProcessorParameters.C3d_parameters = histoMax_C3d_params.clone(
            dR_multicluster_byLayer_coefficientA = distances
            )
    set_histomax_params(producer.ProcessorParameters.C3d_parameters, 0, nBins_R, nBins_Phi, binSumsHisto, seed_threshold)
    return producer


def create_histoInterpolatedMax1stOrder(process, inputs,
                                        distance=histoInterpolatedMax_C3d_params.dR_multicluster,
                                        nBins_R=histoInterpolatedMax_C3d_params.nBins_R_histo_multicluster,
                                        nBins_Phi=histoInterpolatedMax_C3d_params.nBins_Phi_histo_multicluster,
                                        binSumsHisto=histoInterpolatedMax_C3d_params.binSumsHisto,
                                        seed_threshold=histoInterpolatedMax_C3d_params.threshold_histo_multicluster,
                                        ):
    producer = process.hgcalBackEndLayer2Producer.clone(
            InputCluster = cms.InputTag('{}:HGCalBackendLayer1Processor2DClustering'.format(inputs))
            )
    producer.ProcessorParameters.C3d_parameters = histoInterpolatedMax_C3d_params.clone(
            neighbour_weights = neighbour_weights_1stOrder
            )
    set_histomax_params(producer.ProcessorParameters.C3d_parameters, distance, nBins_R, nBins_Phi, binSumsHisto, seed_threshold)
    return producer


def create_histoInterpolatedMax2ndOrder(process, inputs,
                                        distance=histoInterpolatedMax_C3d_params.dR_multicluster,
                                        nBins_R=histoInterpolatedMax_C3d_params.nBins_R_histo_multicluster,
                                        nBins_Phi=histoInterpolatedMax_C3d_params.nBins_Phi_histo_multicluster,
                                        binSumsHisto=histoInterpolatedMax_C3d_params.binSumsHisto,
                                        seed_threshold=histoInterpolatedMax_C3d_params.threshold_histo_multicluster,
                                        ):
    producer = process.hgcalBackEndLayer2Producer.clone(
            InputCluster = cms.InputTag('{}:HGCalBackendLayer1Processor2DClustering'.format(inputs))
            )
    producer.ProcessorParameters.C3d_parameters = histoInterpolatedMax_C3d_params.clone(
            neighbour_weights = neighbour_weights_2ndOrder
            )
    set_histomax_params(producer.ProcessorParameters.C3d_parameters, distance, nBins_R, nBins_Phi, binSumsHisto, seed_threshold)
    return producer


def create_histoThreshold(process, inputs,
                          threshold=histoThreshold_C3d_params.threshold_histo_multicluster,
                          distance=histoThreshold_C3d_params.dR_multicluster,
                          nBins_R=histoThreshold_C3d_params.nBins_R_histo_multicluster,
                          nBins_Phi=histoThreshold_C3d_params.nBins_Phi_histo_multicluster,
                          binSumsHisto=histoThreshold_C3d_params.binSumsHisto
                          ):
    producer = process.hgcalBackEndLayer2Producer.clone(
            InputCluster = cms.InputTag('{}:HGCalBackendLayer1Processor2DClustering'.format(inputs))
            )
    producer.ProcessorParameters.C3d_parameters = histoThreshold_C3d_params.clone()
    set_histomax_params(producer.ProcessorParameters.C3d_parameters, distance, nBins_R, nBins_Phi, binSumsHisto, threshold)
    return producer
