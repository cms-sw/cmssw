import FWCore.ParameterSet.Config as cms

# embedding in photons
cleanPatPhotonsTriggerMatch = cms.EDProducer( "PATTriggerMatchPhotonEmbedder",
    src     = cms.InputTag( "cleanPatPhotons" ),
    matches = cms.VInputTag()
)

# embedding in electrons
cleanPatElectronsTriggerMatch = cms.EDProducer( "PATTriggerMatchElectronEmbedder",
    src     = cms.InputTag( "cleanPatElectrons" ),
    matches = cms.VInputTag( "electronTriggerMatchHltElectrons"
                           , "electronTriggerMatchL1Electrons"
                           )
)

# embedding in muons
cleanPatMuonsTriggerMatch = cms.EDProducer( "PATTriggerMatchMuonEmbedder",
    src     = cms.InputTag( "cleanPatMuons" ),
    matches = cms.VInputTag( "muonTriggerMatchL1Muons"
                           , "muonTriggerMatchAll"
                           , "muonTriggerMatchNone"
                           )
)

# embedding in taus
cleanPatTausTriggerMatch = cms.EDProducer( "PATTriggerMatchTauEmbedder",
    src     = cms.InputTag( "cleanPatTaus" ),
    matches = cms.VInputTag( "tauTriggerMatchTriggerTaus" )
)

# embedding in jets
cleanPatAK5CaloJetsTriggerMatch = cms.EDProducer( "PATTriggerMatchJetEmbedder",
    src     = cms.InputTag( "cleanPatAK5CaloJets" ),
    matches = cms.VInputTag()
)

# embedding in MET
patAK5CaloMETsTriggerMatch = cms.EDProducer( "PATTriggerMatchMETEmbedder",
    src     = cms.InputTag( "patAK5CaloMETs" ),
    matches = cms.VInputTag()
)


# embeding sequence
patTriggerMatchEmbedder = cms.Sequence(
    cleanPatPhotonsTriggerMatch     +
    cleanPatElectronsTriggerMatch   +
    cleanPatMuonsTriggerMatch       +
    cleanPatTausTriggerMatch        +
    cleanPatAK5CaloJetsTriggerMatch +
    patAK5CaloMETsTriggerMatch
)
