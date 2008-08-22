import FWCore.ParameterSet.Config as cms

mcSinglePhotonEtaFilter = cms.EDFilter("MCEtaFilter",
    candTag = cms.InputTag("source"),
    ncandcut = cms.int32(1),
    id = cms.int32(-1)
)



