import FWCore.ParameterSet.Config as cms

eidNeuralNetExt = cms.EDProducer("EleIdNeuralNetExtProducer",
    src = cms.InputTag("gedGsfElectrons"),
    doNeuralNet = cms.bool(True),
    weightsDir = cms.string(''),
    NN_set = cms.string('ZeeZmumuJets-2500ev')
)


