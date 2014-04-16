import FWCore.ParameterSet.Config as cms

slimmedJets = cms.EDProducer("PATJetSlimmer",
   src = cms.InputTag("selectedPatJets"),
   map = cms.InputTag("packedPFCandidates"),
   clearJetVars = cms.bool(True),
   clearDaughters = cms.bool(False),
   clearTrackRefs = cms.bool(True),
   dropSpecific = cms.bool(False),
)
