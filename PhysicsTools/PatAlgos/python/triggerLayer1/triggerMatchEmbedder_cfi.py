import FWCore.ParameterSet.Config as cms

# Embedding in photons
cleanPatPhotonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchPhotonEmbedder"
, src     = cms.InputTag( "cleanPatPhotons" )
, matches = cms.VInputTag(
  )
)
selectedPatPhotonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchPhotonEmbedder"
, src     = cms.InputTag( "selectedPatPhotons" )
, matches = cms.VInputTag(
  )
)

# Embedding in electrons
cleanPatElectronsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchElectronEmbedder"
, src     = cms.InputTag( "cleanPatElectrons" )
, matches = cms.VInputTag(
  )
)
selectedPatElectronsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchElectronEmbedder"
, src     = cms.InputTag( "selectedPatElectrons" )
, matches = cms.VInputTag(
  )
)

# Embedding in muons
cleanPatMuonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMuonEmbedder"
, src     = cms.InputTag( "cleanPatMuons" )
, matches = cms.VInputTag(
  )
)
selectedPatMuonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMuonEmbedder"
, src     = cms.InputTag( "selectedPatMuons" )
, matches = cms.VInputTag(
  )
)

# Embedding in taus
cleanPatTausTriggerMatch = cms.EDProducer(
  "PATTriggerMatchTauEmbedder"
, src     = cms.InputTag( "cleanPatTaus" )
, matches = cms.VInputTag(
  )
)
selectedPatTausTriggerMatch = cms.EDProducer(
  "PATTriggerMatchTauEmbedder"
, src     = cms.InputTag( "selectedPatTaus" )
, matches = cms.VInputTag(
  )
)

# Embedding in jets
cleanPatJetsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJets" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJets" )
, matches = cms.VInputTag(
  )
)

# Embedding in MET
patMETsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMETEmbedder"
, src     = cms.InputTag( "patMETs" )
, matches = cms.VInputTag(
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
