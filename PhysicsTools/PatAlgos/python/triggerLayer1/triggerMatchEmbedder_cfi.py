import FWCore.ParameterSet.Config as cms

# Embedding in photons
cleanPatPhotonsTriggerMatch = cms.EDProducer( "PATTriggerMatchPhotonEmbedder"
, src     = cms.InputTag( "cleanPatPhotons" )
, matches = cms.VInputTag(
  )
)

# Embedding in electrons
cleanPatElectronsTriggerMatch = cms.EDProducer( "PATTriggerMatchElectronEmbedder"
, src     = cms.InputTag( "cleanPatElectrons" )
, matches = cms.VInputTag(
    "electronTriggerMatchHLTEle15LWL1R"
  , "electronTriggerMatchHLTDoubleEle5SWL1R"
  )
)

# Embedding in muons
cleanPatMuonsTriggerMatch = cms.EDProducer( "PATTriggerMatchMuonEmbedder"
, src     = cms.InputTag( "cleanPatMuons" )
, matches = cms.VInputTag(
    "muonTriggerMatchL1Muon"
  , "muonTriggerMatchHLTIsoMu3"
  , "muonTriggerMatchHLTMu3"
  , "muonTriggerMatchHLTDoubleMu3"
  )
)

# Embedding in taus
cleanPatTausTriggerMatch = cms.EDProducer( "PATTriggerMatchTauEmbedder"
, src     = cms.InputTag( "cleanPatTaus" )
, matches = cms.VInputTag(
    "tauTriggerMatchHLTDoubleLooseIsoTau15"
  )
)

# Embedding in jets
cleanPatJetsTriggerMatch = cms.EDProducer( "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJets" )
, matches = cms.VInputTag(
  )
)

# Embedding in MET
patMETsTriggerMatch = cms.EDProducer( "PATTriggerMatchMETEmbedder"
, src     = cms.InputTag( "patMETs" )
, matches = cms.VInputTag(
  )
)


## Embedding sequences
patTriggerMatchEmbedder = cms.Sequence(
  cleanPatPhotonsTriggerMatch
+ cleanPatElectronsTriggerMatch
+ cleanPatMuonsTriggerMatch
+ cleanPatTausTriggerMatch
+ cleanPatJetsTriggerMatch
+ patMETsTriggerMatch
)
