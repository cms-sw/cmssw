import FWCore.ParameterSet.Config as cms

l1GctHwDigis = cms.EDProducer("GctRawToDigi",
    inputLabel = cms.InputTag("source"),
    gctFedId = cms.int32(745),
    hltMode = cms.bool(False),
    grenCompatibilityMode = cms.bool(False),
    unpackRct = cms.untracked.bool(True),
    unpackInternEm = cms.untracked.bool(False),
    unpackInternJets = cms.untracked.bool(False),
    unpackFibres = cms.untracked.bool(False),
    unpackEm = cms.untracked.bool(True),
    unpackJets = cms.untracked.bool(True),
    unpackEtSums = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)


