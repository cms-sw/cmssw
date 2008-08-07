import FWCore.ParameterSet.Config as cms

l1GctHwDigis = cms.EDProducer("GctRawToDigi",
    verbose = cms.untracked.bool(False),
    gctFedId = cms.int32(745),
    hltMode = cms.bool(False),
    grenCompatibilityMode = cms.bool(False),
    inputLabel = cms.InputTag("source"),
    unpackRct = cms.untracked.bool(True),
    unpackEm = cms.untracked.bool(True),
    unpackJets = cms.untracked.bool(True),
    unpackEtSums = cms.untracked.bool(True),
    unpackInternEm = cms.untracked.bool(False),
    unpackFibres = cms.untracked.bool(False)
)


