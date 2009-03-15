import FWCore.ParameterSet.Config as cms

l1GctHwDigis = cms.EDProducer("GctRawToDigi",
    inputLabel = cms.InputTag("source"),
    gctFedId = cms.int32(745),
    hltMode = cms.bool(False),
    unpackerVersion = cms.uint32(0), # O=Auto-detect, or override with: 1=GREN 07 era, 2=CRAFT 08 era, or 3=Apr 2009 onwards.
    verbose = cms.untracked.bool(False)
)


