import FWCore.ParameterSet.Config as cms

rpcTriggerFilter = cms.EDFilter("RPCTriggerFilter",
    DtAndCSC = cms.untracked.bool(False),
    DTTrigger = cms.untracked.bool(False),
    RPCAndDTAndCSC = cms.untracked.bool(False),
    RPCEndcapTrigger = cms.untracked.bool(False),
    EnableTriggerFilter = cms.untracked.bool(True),
    GMTInputTag = cms.untracked.InputTag("gtDigis"),
    RPCAndCSC = cms.untracked.bool(False),
    CSCTrigger = cms.untracked.bool(False),
    RPCBarrelTrigger = cms.untracked.bool(False),
    RPCAndDT = cms.untracked.bool(False),
    RPCTrigger = cms.untracked.bool(True)
)



