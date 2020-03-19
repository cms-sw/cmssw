import FWCore.ParameterSet.Config as cms
from L1Trigger.L1THGCal.hgcalBackEndLayer2Producer_cfi import histoMax_C3d_seeding_params, \
                                                              histoSecondaryMax_C3d_params, \
                                                              histoInterpolatedMax_C3d_params, \
                                                              histoThreshold_C3d_params, \
                                                              histoMaxXYVariableDR_C3d_params, \
                                                              neighbour_weights_1stOrder, \
                                                              neighbour_weights_2ndOrder
                                                              


def set_histomax_seeding_params(parameters_seeding_c3d,
                        nBins_X1,
                        nBins_X2,
                        binSumsHisto,
                        seed_threshold,
                        ):
    parameters_seeding_c3d.nBins_X1_histo_multicluster = nBins_X1
    parameters_seeding_c3d.nBins_X2_histo_multicluster = nBins_X2
    parameters_seeding_c3d.binSumsHisto = binSumsHisto
    parameters_seeding_c3d.threshold_histo_multicluster = seed_threshold


def custom_3dclustering_histoMax(process,
                                 nBins_X1=histoMax_C3d_seeding_params.nBins_X1_histo_multicluster,
                                 nBins_X2=histoMax_C3d_seeding_params.nBins_X2_histo_multicluster,
                                 binSumsHisto=histoMax_C3d_seeding_params.binSumsHisto,
                                 seed_threshold=histoMax_C3d_seeding_params.threshold_histo_multicluster,
                                 seed_position=histoMax_C3d_seeding_params.seed_position,
                                 ):
    parameters_c3d = histoMax_C3d_seeding_params.clone()
    set_histomax_seeding_params(parameters_c3d, nBins_X1, nBins_X2, binSumsHisto,
                        seed_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = parameters_c3d
    return process

def custom_3dclustering_histoSecondaryMax(process,
                                          threshold=histoSecondaryMax_C3d_params.threshold_histo_multicluster,
                                          nBins_X1=histoSecondaryMax_C3d_params.nBins_X1_histo_multicluster,
                                          nBins_X2=histoSecondaryMax_C3d_params.nBins_X2_histo_multicluster,
                                          binSumsHisto=histoSecondaryMax_C3d_params.binSumsHisto,
                                          ):
    parameters_c3d = histoSecondaryMax_C3d_params.clone()
    set_histomax_seeding_params(parameters_c3d, nBins_X1, nBins_X2, binSumsHisto,
                        threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = parameters_c3d
    return process


def custom_3dclustering_histoInterpolatedMax1stOrder(process,
                                                     nBins_X1=histoInterpolatedMax_C3d_params.nBins_X1_histo_multicluster,
                                                     nBins_X2=histoInterpolatedMax_C3d_params.nBins_X2_histo_multicluster,
                                                     binSumsHisto=histoInterpolatedMax_C3d_params.binSumsHisto,
                                                     seed_threshold=histoInterpolatedMax_C3d_params.threshold_histo_multicluster,
                                                     ):
    parameters_c3d = histoInterpolatedMax_C3d_params.clone(
            neighbour_weights = neighbour_weights_1stOrder
            )
    set_histomax_seeding_params(parameters_c3d, nBins_X1, nBins_X2, binSumsHisto,
                        seed_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = parameters_c3d
    return process


def custom_3dclustering_histoInterpolatedMax2ndOrder(process,
                                                     nBins_X1=histoInterpolatedMax_C3d_params.nBins_X1_histo_multicluster,
                                                     nBins_X2=histoInterpolatedMax_C3d_params.nBins_X2_histo_multicluster,
                                                     binSumsHisto=histoInterpolatedMax_C3d_params.binSumsHisto,
                                                     seed_threshold=histoInterpolatedMax_C3d_params.threshold_histo_multicluster,
                                                     ):
    parameters_c3d = histoInterpolatedMax_C3d_params.clone(
            neighbour_weights = neighbour_weights_2ndOrder
            )
    set_histomax_seeding_params(parameters_c3d, nBins_X1, nBins_X2, binSumsHisto,
                        seed_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = parameters_c3d
    return process


def custom_3dclustering_histoThreshold(process,
                                       nBins_X1=histoThreshold_C3d_params.nBins_X1_histo_multicluster,
                                       nBins_X2=histoThreshold_C3d_params.nBins_X2_histo_multicluster,
                                       binSumsHisto=histoThreshold_C3d_params.binSumsHisto,
                                       seed_threshold=histoThreshold_C3d_params.threshold_histo_multicluster,
                                       ):
    parameters_c3d = histoThreshold_C3d_params.clone()
    set_histomax_seeding_params(parameters_c3d, nBins_X1, nBins_X2, binSumsHisto,
                        seed_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = parameters_c3d
    return process


def custom_3dclustering_XYHistoMax(process,
                                   nBins_X1=histoMaxXYVariableDR_C3d_params.nBins_X1_histo_multicluster,
                                   nBins_X2=histoMaxXYVariableDR_C3d_params.nBins_X2_histo_multicluster,
                                   seed_threshold=histoMaxXYVariableDR_C3d_params.threshold_histo_multicluster,
                                   seed_position=histoMaxXYVariableDR_C3d_params.seed_position,
                                   ):
    parameters_c3d = histoMaxXYVariableDR_C3d_params.clone()
    set_histomax_seeding_params(parameters_c3d, nBins_X1, nBins_X2,
            histoMaxXYVariableDR_C3d_params.binSumsHisto,seed_threshold)
    process.hgcalBackEndLayer2Producer.ProcessorParameters.C3d_parameters.histoMax_C3d_seeding_parameters = parameters_c3d
    return process

