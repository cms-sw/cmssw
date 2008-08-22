import FWCore.ParameterSet.Config as cms

mcDoublePhotonEtaFilter = cms.EDFilter("MCEtaFilter",
    candTag = cms.InputTag("source"),
    ncandcut = cms.int32(2),
    id = cms.int32(-1)
)



