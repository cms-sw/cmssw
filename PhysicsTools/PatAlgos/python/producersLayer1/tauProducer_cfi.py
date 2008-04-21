import FWCore.ParameterSet.Config as cms

allLayer1Taus = cms.EDProducer("PATTauProducer",
    addGenMatch = cms.bool(True),
    addResolutions = cms.bool(True),
    isolation = cms.PSet(

    ),
    addLRValues = cms.bool(False),
    isoDeposits = cms.PSet(

    ),
    tauLRFile = cms.string('PhysicsTools/PatUtils/data/TauLRDistros.root'),
    tauSource = cms.InputTag("allLayer0Taus"),
    useNNResolutions = cms.bool(True),
    tauResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_tau.root'),
    genParticleMatch = cms.InputTag("tauMatch")
)


