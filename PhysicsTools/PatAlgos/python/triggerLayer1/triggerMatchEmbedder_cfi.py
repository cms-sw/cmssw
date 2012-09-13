import FWCore.ParameterSet.Config as cms

# Embedding in muons
somePatMuonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMuonEmbedder"
, src     = cms.InputTag( "selectedPatMuons" )
, matches = cms.VInputTag(
    'somePatMuonTriggerMatchHLTMu17'
  , 'somePatMuonTriggerMatchHLTDoubleMu5IsoMu5'
  , 'somePatMuonTriggerMatchHLTMu8DiJet30' # x-trigger
  )
)

# Embedding in photons
somePatPhotonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchPhotonEmbedder"
, src     = cms.InputTag( "selectedPatPhotons" )
, matches = cms.VInputTag(
    'somePatPhotonTriggerMatchHLTPhoton26Photon18'
  )
)

# Embedding in electrons
somePatElectronsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchElectronEmbedder"
, src     = cms.InputTag( "selectedPatElectrons" )
, matches = cms.VInputTag(
    'somePatElectronTriggerMatchHLTEle17CaloIdTCaloIsoVLTrkIdVLTrkIsoVL'
  )
)

# Embedding in taus
somePatTausTriggerMatch = cms.EDProducer(
  "PATTriggerMatchTauEmbedder"
, src     = cms.InputTag( "selectedPatTaus" )
, matches = cms.VInputTag(
    'somePatTauTriggerMatchHLTDoubleMediumIsoPFTau30Trk1eta2p1'
  )
)

# Embedding in jets
somePatJetsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJets" )
, matches = cms.VInputTag(
    'somePatJetTriggerMatchHLTPFJet40'
  , 'somePatJetTriggerMatchHLTMu8DiJet30' # x-trigger
  )
)

# Embedding in MET
somePatMETsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMETEmbedder"
, src     = cms.InputTag( "patMETs" )
, matches = cms.VInputTag(
    'somePatMetTriggerMatchHLTMET120'
  )
)

## Embedding sequences
patTriggerMatchEmbedderDefaultSequence = cms.Sequence(
  somePatPhotonsTriggerMatch
+ somePatElectronsTriggerMatch
+ somePatMuonsTriggerMatch
+ somePatTausTriggerMatch
+ somePatJetsTriggerMatch
+ somePatMETsTriggerMatch
)
