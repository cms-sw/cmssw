import FWCore.ParameterSet.Config as cms

allLayer1Taus = cms.EDProducer("PATTauProducer",
    # MC matching configurables
    addGenMatch = cms.bool(True),
    # resolution configurables
    addResolutions = cms.bool(True),
    # input root file for the resolution functions
    # likelihood ratio configurables
    addLRValues = cms.bool(False),
    tauLRFile = cms.string('PhysicsTools/PatUtils/data/TauLRDistros.root'),
    # General configurables
    tauSource = cms.InputTag("allLayer0Taus"),
    useNNResolutions = cms.bool(True), ## use the neural network approach?

    tauResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_tau.root'),
    genParticleMatch = cms.InputTag("tauMatch") ## particles source to be used for the matching

)


