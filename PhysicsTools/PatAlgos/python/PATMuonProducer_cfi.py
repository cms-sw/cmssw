import FWCore.ParameterSet.Config as cms

allLayer1Muons = cms.EDProducer("PATMuonProducer",
    # MC matching configurables
    addGenMatch = cms.bool(True),
    # Resolution configurables
    addResolutions = cms.bool(True),
    # input root file for the resolution functions
    # Isolation configurables
    doTrkIsolation = cms.bool(True),
    # Likelihood ratio configurables
    addLRValues = cms.bool(True),
    # muon calo-isolation input objects
    # Muon ID configurables
    addMuonID = cms.bool(True),
    doCalIsolation = cms.bool(True), ## switch on/off the calorimeter isolation calculations

    hcalIsoSource = cms.InputTag("muGlobalIsoDepositCalByAssociatorTowers","hcal"),
    useNNResolutions = cms.bool(False), ## use the neural network approach?

    hocalIsoSource = cms.InputTag("muGlobalIsoDepositCalByAssociatorTowers","ho"),
    muonLRFile = cms.string('PhysicsTools/PatUtils/data/MuonLRDistros.root'),
    ecalIsoSource = cms.InputTag("muGlobalIsoDepositCalByAssociatorTowers","ecal"),
    # General configurables
    muonSource = cms.InputTag("allLayer0Muons"),
    tracksSource = cms.InputTag("ctfWithMaterialTracks"), ## tracks to be used for the tracker isolation

    trackIsoSource = cms.InputTag("muGlobalIsoDepositCtfTk"), ## muon track-isolation input object

    muonResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_muon.root'),
    genParticleMatch = cms.InputTag("muonMatch") ## particles source to be used for the matching

)


