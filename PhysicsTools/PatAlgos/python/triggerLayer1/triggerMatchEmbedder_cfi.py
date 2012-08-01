import FWCore.ParameterSet.Config as cms

# Embedding in muons
cleanPatMuonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMuonEmbedder"
, src     = cms.InputTag( "cleanPatMuons" )
, matches = cms.VInputTag(
    'cleanMuonTriggerMatchHLTMu17'
  , 'cleanMuonTriggerMatchHLTDoubleMu5IsoMu5'
  , 'cleanMuonTriggerMatchHLTMu8DiJet30' # x-trigger
  )
)

# Embedding in photons
cleanPatPhotonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchPhotonEmbedder"
, src     = cms.InputTag( "cleanPatPhotons" )
, matches = cms.VInputTag(
    'cleanPhotonTriggerMatchHLTPhoton26Photon18'
  )
)

# Embedding in electrons
cleanPatElectronsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchElectronEmbedder"
, src     = cms.InputTag( "cleanPatElectrons" )
, matches = cms.VInputTag(
    'cleanElectronTriggerMatchHLTEle17CaloIdTCaloIsoVLTrkIdVLTrkIsoVL'
  )
)

# Embedding in taus
cleanPatTausTriggerMatch = cms.EDProducer(
  "PATTriggerMatchTauEmbedder"
, src     = cms.InputTag( "cleanPatTaus" )
, matches = cms.VInputTag(
    'cleanTauTriggerMatchHLTDoubleMediumIsoPFTau30Trk1eta2p1'
  )
)

# Embedding in jets
cleanPatJetsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJets" )
, matches = cms.VInputTag(
    'cleanJetTriggerMatchHLTPFJet40'
  , 'cleanJetTriggerMatchHLTMu8DiJet30' # x-trigger
  )
)

# Embedding in MET
patMETsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMETEmbedder"
, src     = cms.InputTag( "patMETs" )
, matches = cms.VInputTag(
    'metTriggerMatchHLTMET120'
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
