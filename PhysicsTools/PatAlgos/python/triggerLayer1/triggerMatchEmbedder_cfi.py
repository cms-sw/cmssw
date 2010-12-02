import FWCore.ParameterSet.Config as cms

# Embedding in muons
cleanPatMuonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMuonEmbedder"
, src     = cms.InputTag( "cleanPatMuons" )
, matches = cms.VInputTag(
    'cleanMuonTriggerMatchHLTMu9'
  , 'cleanMuonTriggerMatchHLTDoubleIsoMu3'
  )
)

# Embedding in photons
cleanPatPhotonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchPhotonEmbedder"
, src     = cms.InputTag( "cleanPatPhotons" )
, matches = cms.VInputTag(
    'cleanPhotonTriggerMatchHLTPhoton20CleanedL1R'
  )
)

# Embedding in electrons
cleanPatElectronsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchElectronEmbedder"
, src     = cms.InputTag( "cleanPatElectrons" )
, matches = cms.VInputTag(
    'cleanElectronTriggerMatchHLTEle20SWL1R'
  )
)

# Embedding in taus
cleanPatTausTriggerMatch = cms.EDProducer(
  "PATTriggerMatchTauEmbedder"
, src     = cms.InputTag( "cleanPatTaus" )
, matches = cms.VInputTag(
    'cleanTauTriggerMatchHLTDoubleLooseIsoTau15'
  )
)

# Embedding in jets
cleanPatJetsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJets" )
, matches = cms.VInputTag(
    'cleanJetTriggerMatchHLTJet15U'
  )
)

# Embedding in MET
patMETsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMETEmbedder"
, src     = cms.InputTag( "patMETs" )
, matches = cms.VInputTag(
    'metTriggerMatchHLTMET45'
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
