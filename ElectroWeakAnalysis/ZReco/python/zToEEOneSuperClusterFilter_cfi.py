import FWCore.ParameterSet.Config as cms

zToEEOneSuperClusterFilter = cms.EDFilter("CandCountFilter",
    src = cms.InputTag("zToEEOneSuperCluster"),
    minNumber = cms.uint32(1)
)


