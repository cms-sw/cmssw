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
patPhotonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchPhotonEmbedder"
, src     = cms.InputTag( "patPhotons" )
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
patElectronsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchElectronEmbedder"
, src     = cms.InputTag( "patElectrons" )
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
patMuonsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchMuonEmbedder"
, src     = cms.InputTag( "patMuons" )
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
patTausTriggerMatch = cms.EDProducer(
  "PATTriggerMatchTauEmbedder"
, src     = cms.InputTag( "patTaus" )
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
patJetsTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJets" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsIC5CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsIC5Calo" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsIC5CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsIC5Calo" )
, matches = cms.VInputTag(
  )
)
patJetsIC5CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsIC5Calo" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsSC5CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsSC5Calo" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsSC5CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsSC5Calo" )
, matches = cms.VInputTag(
  )
)
patJetsSC5CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsSC5Calo" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsKT4CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsKT4Calo" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsKT4CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsKT4Calo" )
, matches = cms.VInputTag(
  )
)
patJetsKT4CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsKT4Calo" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsKT6CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsKT6Calo" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsKT6CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsKT6Calo" )
, matches = cms.VInputTag(
  )
)
patJetsKT6CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsKT6Calo" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsAK5CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsAK5Calo" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsAK5CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsAK5Calo" )
, matches = cms.VInputTag(
  )
)
patJetsAK5CaloTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsAK5Calo" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsIC5PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsIC5PF" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsIC5PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsIC5PF" )
, matches = cms.VInputTag(
  )
)
patJetsIC5PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsIC5PF" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsSC5PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsSC5PF" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsSC5PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsSC5PF" )
, matches = cms.VInputTag(
  )
)
patJetsSC5PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsSC5PF" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsKT4PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsKT4PF" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsKT4PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsKT4PF" )
, matches = cms.VInputTag(
  )
)
patJetsKT4PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsKT4PF" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsKT6PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsKT6PF" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsKT6PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsKT6PF" )
, matches = cms.VInputTag(
  )
)
patJetsKT6PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsKT6PF" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsAK5PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsAK5PF" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsAK5PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsAK5PF" )
, matches = cms.VInputTag(
  )
)
patJetsAK5PFTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsAK5PF" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsIC5JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsIC5JPT" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsIC5JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsIC5JPT" )
, matches = cms.VInputTag(
  )
)
patJetsIC5JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsIC5JPT" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsSC5JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsSC5JPT" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsSC5JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsSC5JPT" )
, matches = cms.VInputTag(
  )
)
patJetsSC5JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsSC5JPT" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsKT4JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsKT4JPT" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsKT4JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsKT4JPT" )
, matches = cms.VInputTag(
  )
)
patJetsKT4JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsKT4JPT" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsKT6JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsKT6JPT" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsKT6JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsKT6JPT" )
, matches = cms.VInputTag(
  )
)
patJetsKT6JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsKT6JPT" )
, matches = cms.VInputTag(
  )
)
cleanPatJetsAK5JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "cleanPatJetsAK5JPT" )
, matches = cms.VInputTag(
  )
)
selectedPatJetsAK5JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "selectedPatJetsAK5JPT" )
, matches = cms.VInputTag(
  )
)
patJetsAK5JPTTriggerMatch = cms.EDProducer(
  "PATTriggerMatchJetEmbedder"
, src     = cms.InputTag( "patJetsAK5JPT" )
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
