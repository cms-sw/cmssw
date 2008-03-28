import FWCore.ParameterSet.Config as cms

allLayer1METs = cms.EDProducer("PATMETProducer",
    # General configurables
    metSource = cms.InputTag("allLayer0METs"),
    muonSource = cms.InputTag("muons"), ## muon input source for corrections

    # Resolution configurables
    addResolutions = cms.bool(True),
    metResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_met.root'),
    useNNResolutions = cms.bool(False), ## use the neural network approach?

    # MC matching configurables
    addGenMET = cms.bool(True),
    # input root file for the resolution functions
    # Muon correction configurables
    addMuonCorrections = cms.bool(True),
    genParticleSource = cms.InputTag("genParticles") ## particles source to be used for the matching

)


