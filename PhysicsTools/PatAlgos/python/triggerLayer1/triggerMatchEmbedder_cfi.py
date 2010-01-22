import FWCore.ParameterSet.Config as cms

# Embedding in photons
cleanPatPhotonsTriggerMatch = cms.EDProducer( "PATTriggerMatchPhotonEmbedder",
    src     = cms.InputTag( "cleanPatPhotons" ),
    matches = cms.VInputTag(
                           )
)
cleanPatPhotonsTriggerTestMatch = cms.EDProducer( "PATTriggerMatchPhotonEmbedder",
    src     = cms.InputTag( "cleanPatPhotons" ),
    matches = cms.VInputTag(
                           )
)

# Embedding in electrons
cleanPatElectronsTriggerMatch = cms.EDProducer( "PATTriggerMatchElectronEmbedder",
    src     = cms.InputTag( "cleanPatElectrons" ),
    matches = cms.VInputTag( "electronTriggerMatchHLTEle15LWL1R"
                           , "electronTriggerMatchHLTDoubleEle5SWL1R"
                           )
)
cleanPatElectronsTriggerTestMatch = cms.EDProducer( "PATTriggerMatchElectronEmbedder",
    src     = cms.InputTag( "cleanPatElectrons" ),
    matches = cms.VInputTag( "electronTriggerTestMatchHLTElectrons"
                           , "electronTriggerTestMatchHLTFilterEGammas"
                           )
)

# Embedding in muons
cleanPatMuonsTriggerMatch = cms.EDProducer( "PATTriggerMatchMuonEmbedder",
    src     = cms.InputTag( "cleanPatMuons" ),
    matches = cms.VInputTag( "muonTriggerMatchHLTIsoMu3"
                           , "muonTriggerMatchHLTMu3"
                           , "muonTriggerMatchHLTDoubleMu3"
                           )
)
cleanPatMuonsTriggerTestMatch = cms.EDProducer( "PATTriggerMatchMuonEmbedder",
    src     = cms.InputTag( "cleanPatMuons" ),
    matches = cms.VInputTag( "muonTriggerTestMatchL1Muons"
                           , "muonTriggerTestMatchL1CollectionMuons"
                           , "muonTriggerTestMatchNoMuons"
                           )
)

# Embedding in taus
cleanPatTausTriggerMatch = cms.EDProducer( "PATTriggerMatchTauEmbedder",
    src     = cms.InputTag( "cleanPatTaus" ),
    matches = cms.VInputTag( "tauTriggerMatchHLTDoubleLooseIsoTau15"
                           )
)
cleanPatTausTriggerTestMatch = cms.EDProducer( "PATTriggerMatchTauEmbedder",
    src     = cms.InputTag( "cleanPatTaus" ),
    matches = cms.VInputTag(
                           )
)

# Embedding in jets
cleanPatJetsTriggerMatch = cms.EDProducer( "PATTriggerMatchJetEmbedder",
    src     = cms.InputTag( "cleanPatJets" ),
    matches = cms.VInputTag(
                           )
)
cleanPatJetsTriggerTestMatch = cms.EDProducer( "PATTriggerMatchJetEmbedder",
    src     = cms.InputTag( "cleanPatJets" ),
    matches = cms.VInputTag( "jetTriggerTestMatchHLTJet15U"
                           )
)

# Embedding in MET
patMETsTriggerMatch = cms.EDProducer( "PATTriggerMatchMETEmbedder",
    src     = cms.InputTag( "patMETs" ),
    matches = cms.VInputTag(
                           )
)
patMETsTriggerTestMatch = cms.EDProducer( "PATTriggerMatchMETEmbedder",
    src     = cms.InputTag( "patMETs" ),
    matches = cms.VInputTag( "metTriggerTestMatchHLTMET45"
                           , "metTriggerTestMatchHLTMu3"
                           )
)


## Embedding sequences
patTriggerMatchEmbedder = cms.Sequence(
    cleanPatPhotonsTriggerMatch   +
    cleanPatElectronsTriggerMatch +
    cleanPatMuonsTriggerMatch     +
    cleanPatTausTriggerMatch      +
    cleanPatJetsTriggerMatch      +
    patMETsTriggerMatch
)
patTriggerTestMatchEmbedder = cms.Sequence(
    cleanPatPhotonsTriggerTestMatch   +
    cleanPatElectronsTriggerTestMatch +
    cleanPatMuonsTriggerTestMatch     +
    cleanPatTausTriggerTestMatch      +
    cleanPatJetsTriggerTestMatch      +
    patMETsTriggerTestMatch
)
