import FWCore.ParameterSet.Config as cms

gctDigiToRaw = cms.EDProducer("GctDigiToRaw",
    verbose = cms.untracked.bool(False),
    packRctCalo = cms.untracked.bool(True),
    gctFedId = cms.int32(745),
    packRctEm = cms.untracked.bool(True),
    rctInputLabel = cms.InputTag("simRctDigis"),
    gctInputLabel = cms.InputTag("simGctDigis")
)


