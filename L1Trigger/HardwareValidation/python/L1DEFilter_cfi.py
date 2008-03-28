import FWCore.ParameterSet.Config as cms

l1defilter = cms.EDFilter("L1DEFilter",
    FlagSystems = cms.untracked.vuint32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    DataEmulCompareSource = cms.InputTag("l1Compare")
)


