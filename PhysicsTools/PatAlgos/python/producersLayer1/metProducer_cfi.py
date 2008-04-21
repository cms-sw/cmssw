import FWCore.ParameterSet.Config as cms

allLayer1METs = cms.EDProducer("PATMETProducer",
    metSource = cms.InputTag("allLayer0METs"),
    muonSource = cms.InputTag("muons"),
    addResolutions = cms.bool(True),
    metResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_met.root'),
    genMETSource = cms.InputTag("genMet"),
    useNNResolutions = cms.bool(False),
    addGenMET = cms.bool(True),
    addMuonCorrections = cms.bool(True)
)


