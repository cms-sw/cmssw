import FWCore.ParameterSet.Config as cms

# embedding in photons
cleanLayer1PhotonsTriggerMatch = cms.EDProducer( "PATTriggerMatchPhotonEmbedder",
    src     = cms.InputTag( "cleanLayer1Photons" ),
    matches = cms.VInputTag()
)

# embedding in electrons
cleanLayer1ElectronsTriggerMatch = cms.EDProducer( "PATTriggerMatchElectronEmbedder",
    src     = cms.InputTag( "cleanLayer1Electrons" ),
    matches = cms.VInputTag( "electronTriggerMatchHltElectrons"
                           , "electronTriggerMatchL1Electrons"
                           )
)

# embedding in muons
cleanLayer1MuonsTriggerMatch = cms.EDProducer( "PATTriggerMatchMuonEmbedder",
    src     = cms.InputTag( "cleanLayer1Muons" ),
    matches = cms.VInputTag( "muonTriggerMatchL1Muons"
                           , "muonTriggerMatchAll"
                           , "muonTriggerMatchNone"
                           )
)

# embedding in taus
cleanLayer1TausTriggerMatch = cms.EDProducer( "PATTriggerMatchTauEmbedder",
    src     = cms.InputTag( "cleanLayer1Taus" ),
    matches = cms.VInputTag( "tauTriggerMatchTriggerTaus" )
)

# embedding in jets
cleanLayer1JetsTriggerMatch = cms.EDProducer( "PATTriggerMatchJetEmbedder",
    src     = cms.InputTag( "cleanLayer1Jets" ),
    matches = cms.VInputTag()
)

# embedding in MET
layer1METsTriggerMatch = cms.EDProducer( "PATTriggerMatchMETEmbedder",
    src     = cms.InputTag( "layer1METs" ),
    matches = cms.VInputTag()
)


# embeding sequence
patTriggerMatchEmbedder = cms.Sequence(
    cleanLayer1PhotonsTriggerMatch   +
    cleanLayer1ElectronsTriggerMatch +
    cleanLayer1MuonsTriggerMatch     +
    cleanLayer1TausTriggerMatch      +
    cleanLayer1JetsTriggerMatch      +
    layer1METsTriggerMatch
)
