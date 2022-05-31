import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.pfnInPath import pfnInPath
import os

from RecoMET.METPUSubtraction.deepMETProducer_cfi import deepMETProducer as _deepMETProducer

deepMETsResolutionTune = _deepMETProducer.clone()
deepMETsResponseTune = _deepMETProducer.clone(
    graph_path = 'RecoMET/METPUSubtraction/data/models/deepmet/deepmet_resp_v1_2018/model.graphdef',
)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(
    deepMETsResolutionTune,
    max_n_pf=12500,
    graph_path="RecoMET/METPUSubtraction/data/models/deepmet_phase2/deepmet_v1_phase2/model.graphdef"
)
phase2_common.toModify(
    deepMETsResponseTune,
    max_n_pf=12500,
    graph_path="RecoMET/METPUSubtraction/data/models/deepmet_phase2/deepmet_resp_v1_phase2/model.graphdef"
)

from Configuration.Eras.Modifier_run2_jme_2016_cff import run2_jme_2016
run2_jme_2016.toModify(
    deepMETsResponseTune,
    graph_path="RecoMET/METPUSubtraction/data/models/deepmet/deepmet_resp_v1_2016/model.graphdef"
)

from RecoMET.METPUSubtraction.deepMETSonicProducer_cff import deepMETSonicProducer as _deepMETSonicProducer
from Configuration.ProcessModifiers.deepMETSonicTriton_cff import deepMETSonicTriton

# propagate possible parameter updates
_deepMETsResolutionTuneSonic = _deepMETSonicProducer.clone(
    max_n_pf = deepMETsResolutionTune.max_n_pf,
    Client = dict(
        modelName = deepMETsResolutionTune.graph_path.value().split('/')[-3],
        modelConfigPath = '/'.join(deepMETsResolutionTune.graph_path.value().split('/')[:-2])+'/config.pbtxt',
        modelVersion = os.path.realpath(pfnInPath(deepMETsResolutionTune.graph_path.value()).split(':')[-1]).split('/')[-2], #model "1"
    ),
)
deepMETSonicTriton.toReplaceWith(deepMETsResolutionTune, _deepMETsResolutionTuneSonic)

_deepMETsResponseTuneSonic = _deepMETSonicProducer.clone(
    max_n_pf = deepMETsResponseTune.max_n_pf,
    Client = dict(
        modelName = deepMETsResponseTune.graph_path.value().split('/')[-3],
        modelConfigPath = '/'.join(deepMETsResponseTune.graph_path.value().split('/')[:-2])+'/config.pbtxt',
        modelVersion = os.path.realpath(pfnInPath(deepMETsResponseTune.graph_path.value()).split(':')[-1]).split('/')[-2], #model "2"
    ),
)
deepMETSonicTriton.toReplaceWith(deepMETsResponseTune, _deepMETsResponseTuneSonic)

