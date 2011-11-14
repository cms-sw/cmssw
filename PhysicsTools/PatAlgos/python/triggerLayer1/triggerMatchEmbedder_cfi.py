import FWCore.ParameterSet.Config as cms

# Embedding in muons
cleanPatMuonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMuonEmbedder"
, src     = cms.InputTag( "cleanPatMuons" )
, matches = cms.VInputTag(
    'cleanMuonTriggerMatchHLTMu20'
  , 'cleanMuonTriggerMatchHLTDoubleMu6'
  )
)

# Embedding in photons
cleanPatPhotonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchPhotonEmbedder"
, src     = cms.InputTag( "cleanPatPhotons" )
, matches = cms.VInputTag(
    'cleanPhotonTriggerMatchHLTPhoton26IsoVLPhoton18'
  )
)

# Embedding in electrons
cleanPatElectronsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchElectronEmbedder"
, src     = cms.InputTag( "cleanPatElectrons" )
, matches = cms.VInputTag(
    'cleanElectronTriggerMatchHLTEle27CaloIdVTCaloIsoTTrkIdTTrkIsoT'
  )
)

# Embedding in taus
cleanPatTausTriggerMatch = cms.EDProducer(
  "PATTriggerMatchTauEmbedder"
, src     = cms.InputTag( "cleanPatTaus" )
, matches = cms.VInputTag(
    'cleanTauTriggerMatchHLTDoubleIsoPFTau20Trk5'
  )
)

# Embedding in jets
cleanPatJetsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJets" )
, matches = cms.VInputTag(
    'cleanJetTriggerMatchHLTJet240'
  )
)

# Embedding in MET
patMETsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMETEmbedder"
, src     = cms.InputTag( "patMETs" )
, matches = cms.VInputTag(
    'metTriggerMatchHLTMET100'
  )
)

## Embedding sequences
patTriggerMatchEmbedderDefaultSequence = cms.Sequence(
  cleanPatPhotonsTriggerMatch
+ cleanPatElectronsTriggerMatch
+ cleanPatMuonsTriggerMatch
+ cleanPatTausTriggerMatch
+ cleanPatJetsTriggerMatch
+ patMETsTriggerMatch
)
