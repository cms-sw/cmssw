import FWCore.ParameterSet.Config as cms

eidNeuralNetExt = cms.EDProducer("EleIdNeuralNetExtProducer",
    src = cms.InputTag("pixelMatchGsfElectrons"),
    doNeuralNet = cms.bool(True),
    filter = cms.bool(False),
    threshold = cms.double(0.5),
    
    weightsDir = cms.string(''),
    NN_set = cms.string('ZeeZmumuJets-2500ev')
)


