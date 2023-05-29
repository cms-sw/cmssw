import FWCore.ParameterSet.Config as cms

from RecoMET.METPUSubtraction.deepMETSonicProducer_cfi import deepMETSonicProducer as _deepMETSonicProducer

deepMETSonicProducer = _deepMETSonicProducer.clone(
    Client = dict(
        timeout = 300,
        mode = "Async",
        modelName = "deepmet",
        modelConfigPath = "RecoMET/METPUSubtraction/data/models/deepmet/config.pbtxt",
        # version "1" is the resolutionTune
        # version "2" is the responeTune
        modelVersion = "1",
    ),
    pf_src = "packedPFCandidates",
)
