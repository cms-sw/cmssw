import FWCore.ParameterSet.Config as cms

slimmedTaus = cms.EDProducer("PATTauSlimmer",
   src = cms.InputTag("selectedPatTaus"),
   linkToPackedPFCandidates = cms.bool(True),
   packedPFCandidates = cms.InputTag("packedPFCandidates"), 
)

