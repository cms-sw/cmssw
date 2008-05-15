import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.neuralNetElectronId_cfi import *
eidNeuralNet = cms.EDFilter("EleIdNeuralNetRef",
    NeuralNet,
    src = cms.InputTag("pixelMatchGsfElectrons"),
    doNeuralNet = cms.bool(True),
    endcapClusterShapeAssociation = cms.InputTag("islandBasicClusters","islandEndcapShapeAssoc"),
    filter = cms.bool(False),
    barrelClusterShapeAssociation = cms.InputTag("hybridSuperClusters","hybridShapeAssoc"),
    threshold = cms.double(0.5)
)


