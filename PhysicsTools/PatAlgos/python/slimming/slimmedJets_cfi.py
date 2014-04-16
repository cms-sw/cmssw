import FWCore.ParameterSet.Config as cms

slimmedJets = cms.EDProducer("PATJetSlimmer",
   src = cms.InputTag("selectedPatJets"),
   clearJetVars = cms.bool(True),
   clearDaughters = cms.bool(False),
   clearTrackRefs = cms.bool(True),
   dropSpecific = cms.bool(False),
)
