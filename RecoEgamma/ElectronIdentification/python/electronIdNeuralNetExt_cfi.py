import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.neuralNetElectronId_cfi import *
eidNeuralNetExt = cms.EDProducer("EleIdNeuralNetExtProducer",
    NeuralNet,
    src = cms.InputTag("pixelMatchGsfElectrons"),
    doNeuralNet = cms.bool(True),
    filter = cms.bool(False),
    threshold = cms.double(0.5)
)


