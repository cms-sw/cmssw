import FWCore.ParameterSet.Config as cms

from RecoEcal.EgammaClusterProducers.SCEnergyCorrectorDRNProducer_cfi import SCEnergyCorrectorDRNProducer as _SCEnergyCorrectorDRNProducer

DRNProducerEB = _SCEnergyCorrectorDRNProducer.clone(
    inputSCs = "particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel",
    Client = dict(
        modelName = "MustacheEB",
        modelConfigPath = "RecoEcal/EgammaClusterProducers/data/models/MustacheEB/config.pbtxt",
        timeout = 10,
    ),
)

DRNProducerEE = _SCEnergyCorrectorDRNProducer.clone(
    inputSCs = "particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower",
    Client = dict(
        modelName = "MustacheEE",
        modelConfigPath = "RecoEcal/EgammaClusterProducers/data/models/MustacheEE/config.pbtxt",
        timeout = 10,
    ),
)
