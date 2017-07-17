import FWCore.ParameterSet.Config as cms

slimmedTaus = cms.EDProducer("PATTauSlimmer",
   src = cms.InputTag("selectedPatTaus"),
   linkToPackedPFCandidates = cms.bool(True),
   dropPiZeroRefs = cms.bool(True),
   dropTauChargedHadronRefs = cms.bool(True),
   dropPFSpecific = cms.bool(True),
   packedPFCandidates = cms.InputTag("packedPFCandidates"), 
   modifyTaus = cms.bool(True),
   modifierConfig = cms.PSet( modifications = cms.VPSet() )    
)

