import FWCore.ParameterSet.Config as cms

mcSingleElectronEtaFilter = cms.EDFilter("MCEtaFilter",
    candTag = cms.InputTag("source"),
    ncandcut = cms.int32(1),
    id = cms.int32(-1)
)



