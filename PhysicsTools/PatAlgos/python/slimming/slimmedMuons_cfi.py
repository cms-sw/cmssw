import FWCore.ParameterSet.Config as cms

slimmedMuons = cms.EDProducer("PATMuonSlimmer",
    src = cms.InputTag("selectedPatMuons"),
    linkToPackedPFCandidates = cms.bool(True),
    pfCandidates = cms.InputTag("particleFlow"),
    packedPFCandidates = cms.InputTag("packedPFCandidates"), 
)

