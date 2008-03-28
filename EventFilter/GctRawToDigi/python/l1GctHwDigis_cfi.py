import FWCore.ParameterSet.Config as cms

l1GctHwDigis = cms.EDFilter("GctRawToDigi",
    unpackEm = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    inputLabel = cms.InputTag("source"),
    unpackFibres = cms.untracked.bool(False),
    grenCompatibilityMode = cms.bool(False),
    gctFedId = cms.int32(745),
    unpackInternEm = cms.untracked.bool(False),
    unpackJets = cms.untracked.bool(True),
    unpackRct = cms.untracked.bool(True),
    hltMode = cms.bool(False),
    unpackEtSums = cms.untracked.bool(True)
)


