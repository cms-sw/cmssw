import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import dummy_C2d_params, \
                                                              distance_C2d_params, \
                                                              topological_C2d_params, \
                                                              constrTopological_C2d_params
from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import distance_C3d_params, \
                                                              dbscan_C3d_params, \
                                                              histoMax_C3d_params, \
                                                              histoMaxVariableDR_C3d_params, \
                                                              histoSecondaryMax_C3d_params, \
                                                              histoInterpolatedMax_C3d_params, \
                                                              histoThreshold_C3d_params, \
                                                              dr_layerbylayer, \
                                                              dr_layerbylayer_Bcoefficient, \
                                                              neighbour_weights_1stOrder, \
                                                              neighbour_weights_2ndOrder, \
                                                              MAX_LAYERS


def set_threshold_params(pset, seed_threshold, cluster_threshold):
    pset.seeding_threshold_silicon = seed_threshold
    pset.seeding_threshold_scintillator = seed_threshold
    pset.clustering_threshold_silicon = cluster_threshold
    pset.clustering_threshold_scintillator = cluster_threshold


def custom_2dclustering_distance(process,
                                 distance=distance_C2d_params.dR_cluster,  # cm
                                 seed_threshold=distance_C2d_params.seeding_threshold_silicon,  # MipT
                                 cluster_threshold=distance_C2d_params.clustering_threshold_silicon  # MipT
                                 ):
    parameters_c2d = distance_C2d_params.clone(dR_cluster = distance)
    set_threshold_params(parameters_c2d, seed_threshold, cluster_threshold)
    process.hgcalBackEndLayer1Producer.ProcessorParameters.C2d_parameters = parameters_c2d
    return process


def custom_2dclustering_topological(process,
                                    seed_threshold=topological_C2d_params.seeding_threshold_silicon,  # MipT
                                    cluster_threshold=topological_C2d_params.clustering_threshold_silicon  # MipT
                                    ):
    parameters_c2d = topological_C2d_params.clone()
    set_threshold_params(parameters_c2d, seed_threshold, cluster_threshold)
    process.hgcalBackEndLayer1Producer.ProcessorParameters.C2d_parameters = parameters_c2d
    return process


def custom_2dclustering_constrainedtopological(process,
                                               distance=constrTopological_C2d_params.dR_cluster,  # cm
                                               seed_threshold=constrTopological_C2d_params.seeding_threshold_silicon,  # MipT
                                               cluster_threshold=constrTopological_C2d_params.clustering_threshold_silicon  # MipT
                                               ):
    parameters_c2d = constrTopological_C2d_params.clone(dR_cluster = distance)
    set_threshold_params(parameters_c2d, seed_threshold, cluster_threshold)
    process.hgcalBackEndLayer1Producer.ProcessorParameters.C2d_parameters = parameters_c2d
    return process


def custom_2dclustering_dummy(process):
    process.hgcalBackEndLayer1Producer.ProcessorParameters.C2d_parameters = dummy_C2d_params.clone()
    return process


def custom_3dclustering_distance(process,
                                 distance=distance_C3d_params.dR_multicluster
                                 ):
    parameters_c3d = distance_C3d_params.clone(dR_multicluster = distance)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters = parameters_c3d
    return process


def custom_3dclustering_dbscan(process,
                               distance=dbscan_C3d_params.dist_dbscan_multicluster,
                               min_points=dbscan_C3d_params.minN_dbscan_multicluster
                               ):
    parameters_c3d = dbscan_C3d_params.clone(
            dist_dbscan_multicluster = distance,
            minN_dbscan_multicluster = min_points
            )
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters = parameters_c3d
    return process


def set_histomax_params(parameters_c3d,
                        distance,
                        nBins_R,
                        nBins_Phi,
                        binSumsHisto,
                        seed_threshold,
                        shape_threshold
                        ):
    parameters_c3d.dR_multicluster = distance
    parameters_c3d.nBins_R_histo_multicluster = nBins_R
    parameters_c3d.nBins_Phi_histo_multicluster = nBins_Phi
    parameters_c3d.binSumsHisto = binSumsHisto
    parameters_c3d.threshold_histo_multicluster = seed_threshold
    parameters_c3d.shape_threshold = shape_threshold


def custom_3dclustering_histoMax(process,
                                 distance=histoMax_C3d_params.dR_multicluster,
                                 nBins_R=histoMax_C3d_params.nBins_R_histo_multicluster,
                                 nBins_Phi=histoMax_C3d_params.nBins_Phi_histo_multicluster,
                                 binSumsHisto=histoMax_C3d_params.binSumsHisto,
                                 seed_threshold=histoMax_C3d_params.threshold_histo_multicluster,
                                 shape_threshold=histoMax_C3d_params.shape_threshold,
                                 ):
    parameters_c3d = histoMax_C3d_params.clone()
    set_histomax_params(parameters_c3d, distance, nBins_R, nBins_Phi, binSumsHisto,
                        seed_threshold, shape_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters = parameters_c3d
    return process


def custom_3dclustering_histoSecondaryMax(process,
                                          distance=histoSecondaryMax_C3d_params.dR_multicluster,
                                          threshold=histoSecondaryMax_C3d_params.threshold_histo_multicluster,
                                          nBins_R=histoSecondaryMax_C3d_params.nBins_R_histo_multicluster,
                                          nBins_Phi=histoSecondaryMax_C3d_params.nBins_Phi_histo_multicluster,
                                          binSumsHisto=histoSecondaryMax_C3d_params.binSumsHisto,
                                          shape_threshold=histoSecondaryMax_C3d_params.shape_threshold,
                                          ):
    parameters_c3d = histoSecondaryMax_C3d_params.clone()
    set_histomax_params(parameters_c3d, distance, nBins_R, nBins_Phi, binSumsHisto,
                        threshold, shape_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters = parameters_c3d
    return process


def custom_3dclustering_histoMax_variableDr(process,
                                            distances=histoMaxVariableDR_C3d_params.dR_multicluster_byLayer_coefficientA,
                                            nBins_R=histoMaxVariableDR_C3d_params.nBins_R_histo_multicluster,
                                            nBins_Phi=histoMaxVariableDR_C3d_params.nBins_Phi_histo_multicluster,
                                            binSumsHisto=histoMaxVariableDR_C3d_params.binSumsHisto,
                                            seed_threshold=histoMaxVariableDR_C3d_params.threshold_histo_multicluster,
                                            seed_position=histoMaxVariableDR_C3d_params.seed_position,
                                            shape_threshold=histoMaxVariableDR_C3d_params.shape_threshold,
                                            ):
    parameters_c3d = histoMaxVariableDR_C3d_params.clone(
            dR_multicluster_byLayer_coefficientA = cms.vdouble(distances)
            )
    set_histomax_params(parameters_c3d, 0, nBins_R, nBins_Phi, binSumsHisto,
                        seed_threshold, shape_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters = parameters_c3d
    return process


def custom_3dclustering_histoInterpolatedMax1stOrder(process,
                                                     distance=histoInterpolatedMax_C3d_params.dR_multicluster,
                                                     nBins_R=histoInterpolatedMax_C3d_params.nBins_R_histo_multicluster,
                                                     nBins_Phi=histoInterpolatedMax_C3d_params.nBins_Phi_histo_multicluster,
                                                     binSumsHisto=histoInterpolatedMax_C3d_params.binSumsHisto,
                                                     seed_threshold=histoInterpolatedMax_C3d_params.threshold_histo_multicluster,
                                                     shape_threshold=histoInterpolatedMax_C3d_params.shape_threshold,
                                                     ):
    parameters_c3d = histoInterpolatedMax_C3d_params.clone(
            neighbour_weights = neighbour_weights_1stOrder
            )
    set_histomax_params(parameters_c3d, distance, nBins_R, nBins_Phi, binSumsHisto,
                        seed_threshold, shape_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters = parameters_c3d
    return process


def custom_3dclustering_histoInterpolatedMax2ndOrder(process,
                                                     distance=histoInterpolatedMax_C3d_params.dR_multicluster,
                                                     nBins_R=histoInterpolatedMax_C3d_params.nBins_R_histo_multicluster,
                                                     nBins_Phi=histoInterpolatedMax_C3d_params.nBins_Phi_histo_multicluster,
                                                     binSumsHisto=histoInterpolatedMax_C3d_params.binSumsHisto,
                                                     seed_threshold=histoInterpolatedMax_C3d_params.threshold_histo_multicluster,
                                                     shape_threshold=histoInterpolatedMax_C3d_params.shape_threshold,
                                                     ):
    parameters_c3d = histoInterpolatedMax_C3d_params.clone(
            neighbour_weights = neighbour_weights_2ndOrder
            )
    set_histomax_params(parameters_c3d, distance, nBins_R, nBins_Phi, binSumsHisto,
                        seed_threshold, shape_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters = parameters_c3d
    return process


def custom_3dclustering_histoThreshold(process,
                                       distance=histoThreshold_C3d_params.dR_multicluster,
                                       nBins_R=histoThreshold_C3d_params.nBins_R_histo_multicluster,
                                       nBins_Phi=histoThreshold_C3d_params.nBins_Phi_histo_multicluster,
                                       binSumsHisto=histoThreshold_C3d_params.binSumsHisto,
                                       seed_threshold=histoThreshold_C3d_params.threshold_histo_multicluster,
                                       shape_threshold=histoThreshold_C3d_params.shape_threshold,
                                       ):
    parameters_c3d = histoThreshold_C3d_params.clone()
    set_histomax_params(parameters_c3d, distance, nBins_R, nBins_Phi, binSumsHisto,
                        seed_threshold, shape_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters = parameters_c3d
    return process


def custom_3dclustering_clusteringRadiusLayerbyLayerVariableEta(process,
                                                                distance_coefficientA=dr_layerbylayer,
                                                                distance_coefficientB=dr_layerbylayer_Bcoefficient):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.dR_multicluster_byLayer_coefficientA = distance_coefficientA
    parameters_c3d.dR_multicluster_byLayer_coefficientB = distance_coefficientB
    return process


def custom_3dclustering_clusteringRadiusLayerbyLayerFixedEta(process,
                                                             distance_coefficientA=dr_layerbylayer):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.dR_multicluster_byLayer_coefficientA = distance_coefficientA
    parameters_c3d.dR_multicluster_byLayer_coefficientB = cms.vdouble( [0]*(MAX_LAYERS+1) )
    return process

def custom_3dclustering_clusteringRadiusNoLayerDependenceFixedEta(process,
                                                                  distance_coefficientA=0.03):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.dR_multicluster_byLayer_coefficientA = cms.vdouble( [distance_coefficientA]*(MAX_LAYERS+1) )
    parameters_c3d.dR_multicluster_byLayer_coefficientB = cms.vdouble( [0]*(MAX_LAYERS+1) )
    return process

def custom_3dclustering_clusteringRadiusNoLayerDependenceVariableEta(process,
                                                                     distance_coefficientA=0.03,
                                                                     distance_coefficientB=0.02):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.dR_multicluster_byLayer_coefficientA = cms.vdouble( [distance_coefficientA]*(MAX_LAYERS+1) )
    parameters_c3d.dR_multicluster_byLayer_coefficientB = cms.vdouble( [distance_coefficientB]*(MAX_LAYERS+1) )
    return process


def custom_3dclustering_nearestNeighbourAssociation(process):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.cluster_association = cms.string('NearestNeighbour')

    return process

def custom_3dclustering_EnergySplitAssociation(process):

    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    parameters_c3d.cluster_association = cms.string('EnergySplit')
    return process
