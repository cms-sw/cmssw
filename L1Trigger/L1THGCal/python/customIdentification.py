import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.egammaIdentification import *

def custom_identification_drnn_cone(process,
        working_points=tight_wp
        ):
    if len(working_points)!=len(working_points_drnn_cone):
        raise RuntimeError('HGC TPG ID: Number of working points ({0}) not compatible with number of categories ({1})'.format(
                    len(working_points), len(working_points_drnn_cone)))
    for wp,cat in zip(working_points,working_points_drnn_cone):
        if not wp in cat:
            raise KeyError('HGC TPG ID: Cannot find a cut corresponding to the working point {}'.format(wp))
    parameters_c3d = process.l1tHGCalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    identification = egamma_identification_drnn_cone.clone(
            WorkingPoints = cms.vdouble(
                [wps[eff] for wps,eff in zip(working_points_drnn_cone,working_points)]
                )
            )
    parameters_c3d.EGIdentification = identification
    return process


def custom_identification_drnn_dbscan(process,
        working_points=tight_wp
        ):
    if len(working_points)!=len(working_points_drnn_dbscan):
        raise RuntimeError('HGC TPG ID: Number of working points ({0}) not compatible with number of categories ({1})'.format(
                    len(working_points), len(working_points_drnn_dbscan)))
    for wp,cat in zip(working_points,working_points_drnn_dbscan):
        if not wp in cat:
            raise KeyError('HGC TPG ID: Cannot find a cut corresponding to the working point {}'.format(wp))
    parameters_c3d = process.l1tHGCalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    identification = egamma_identification_drnn_dbscan.clone(
            WorkingPoints = cms.vdouble(
                [wps[eff] for wps,eff in zip(working_points_drnn_dbscan,working_points)]
                )
            )
    parameters_c3d.EGIdentification = identification
    return process


def custom_identification_histomax(process,
        working_points=tight_wp
        ):
    if len(working_points)!=len(working_points_histomax):
        raise RuntimeError('HGC TPG ID: Number of working points ({0}) not compatible with number of categories ({1})'.format(
                    len(working_points), len(working_points_histomax)))
    for wp,cat in zip(working_points,working_points_drnn_dbscan):
        if not wp in cat:
            raise KeyError('HGC TPG ID: Cannot find a cut corresponding to the working point {}'.format(wp))
    parameters_c3d = process.l1tHGCalBackEndLayer2Producer.ProcessorParameters.C3d_parameters
    identification = egamma_identification_histomax.clone(
            WorkingPoints = cms.vdouble(
                [wps[eff] for wps,eff in zip(working_points_histomax,working_points)]
                )
            )
    parameters_c3d.EGIdentification = identification
    return process
