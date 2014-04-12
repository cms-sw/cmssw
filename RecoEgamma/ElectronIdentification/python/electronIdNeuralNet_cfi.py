import FWCore.ParameterSet.Config as cms

eidNeuralNet = cms.EDFilter("EleIdNeuralNetRef",
    filter = cms.bool(False),
    threshold = cms.double(0.5),
    src = cms.InputTag("gedGsfElectrons"),
    doNeuralNet = cms.bool(True),
    weightsDir = cms.string(''),
    NN_set = cms.string('ZeeZmumuJets-2500ev')
)


