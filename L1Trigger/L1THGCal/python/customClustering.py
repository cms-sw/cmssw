import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import dummy_C2d_params, \
                                                              distance_C2d_params, \
                                                              topological_C2d_params, \
                                                              constrTopological_C2d_params
from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import distance_C3d_params, \
                                                              dbscan_C3d_params, \
                                                              histoMax_C3d_clustering_params, \
                                                              histoMaxVariableDR_C3d_params, \
                                                              histoMaxXYVariableDR_C3d_params, \
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


def set_histomax_clustering_params(parameters_c3d,
                        distance,
                        shape_threshold,
                        shape_distance
                        ):
    parameters_c3d.dR_multicluster = distance
    parameters_c3d.shape_threshold = shape_threshold
    parameters_c3d.shape_distance = shape_distance


def custom_3dclustering_fixedRadius(process,
                                distance=histoMax_C3d_clustering_params.dR_multicluster,
                                shape_threshold=histoMax_C3d_clustering_params.shape_threshold,
                                shape_distance=histoMax_C3d_clustering_params.shape_distance
                                ):
    parameters_c3d = histoMax_C3d_clustering_params.clone()
    set_histomax_clustering_params(parameters_c3d, distance, shape_threshold, shape_distance)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters = parameters_c3d
    return process


def custom_3dclustering_variableDr(process, distances=histoMaxVariableDR_C3d_params.dR_multicluster_byLayer_coefficientA,
                                            shape_threshold=histoMaxVariableDR_C3d_params.shape_threshold,
                                            shape_distance=histoMaxVariableDR_C3d_params.shape_distance
                                            ):
    parameters_c3d = histoMaxVariableDR_C3d_params.clone(
            dR_multicluster_byLayer_coefficientA = cms.vdouble(distances)
            )
    set_histomax_clustering_params(parameters_c3d, 0, shape_threshold, shape_distance)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters = parameters_c3d
    return process


def custom_3dclustering_clusteringRadiusLayerbyLayerVariableEta(process,
                                                                distance_coefficientA=dr_layerbylayer,
                                                                distance_coefficientB=dr_layerbylayer_Bcoefficient):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters
    parameters_c3d.dR_multicluster_byLayer_coefficientA = distance_coefficientA
    parameters_c3d.dR_multicluster_byLayer_coefficientB = distance_coefficientB
    return process


def custom_3dclustering_clusteringRadiusLayerbyLayerFixedEta(process,
                                                             distance_coefficientA=dr_layerbylayer):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters
    parameters_c3d.dR_multicluster_byLayer_coefficientA = distance_coefficientA
    parameters_c3d.dR_multicluster_byLayer_coefficientB = cms.vdouble( [0]*(MAX_LAYERS+1) )
    return process

def custom_3dclustering_clusteringRadiusNoLayerDependenceFixedEta(process,
                                                                  distance_coefficientA=0.03):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters
    parameters_c3d.dR_multicluster_byLayer_coefficientA = cms.vdouble( [distance_coefficientA]*(MAX_LAYERS+1) )
    parameters_c3d.dR_multicluster_byLayer_coefficientB = cms.vdouble( [0]*(MAX_LAYERS+1) )
    return process

def custom_3dclustering_clusteringRadiusNoLayerDependenceVariableEta(process,
                                                                     distance_coefficientA=0.03,
                                                                     distance_coefficientB=0.02):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters
    parameters_c3d.dR_multicluster_byLayer_coefficientA = cms.vdouble( [distance_coefficientA]*(MAX_LAYERS+1) )
    parameters_c3d.dR_multicluster_byLayer_coefficientB = cms.vdouble( [distance_coefficientB]*(MAX_LAYERS+1) )
    return process


def custom_3dclustering_nearestNeighbourAssociation(process):
    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters
    parameters_c3d.cluster_association = cms.string('NearestNeighbour')

    return process

def custom_3dclustering_EnergySplitAssociation(process):

    parameters_c3d = process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_clustering_parameters
    parameters_c3d.cluster_association = cms.string('EnergySplit')
    return process
