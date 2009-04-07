import FWCore.ParameterSet.Config as cms

l1GctHwDigis = cms.EDProducer("GctRawToDigi",
    inputLabel = cms.InputTag("source"),
    gctFedId = cms.int32(745),
    hltMode = cms.bool(False),
    unpackerVersion = cms.uint32(0), # O=Auto-detect, or override with: 1=MCLegacy, 2=V35
    verbose = cms.untracked.bool(False)
)


