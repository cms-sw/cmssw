import FWCore.ParameterSet.Config as cms

l1defilter = cms.EDFilter("L1DEFilter",
    DataEmulCompareSource = cms.InputTag("l1Compare"),
    FlagSystems = cms.untracked.vuint32(
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    # ETP,HTP,RCT,GCT,DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT
    )
)


