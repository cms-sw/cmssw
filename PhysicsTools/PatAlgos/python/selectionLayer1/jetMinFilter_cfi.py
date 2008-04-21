import FWCore.ParameterSet.Config as cms

minLayer1Jets = cms.EDFilter("PATJetMinFilter",
    src = cms.InputTag("selectedLayer1Jets"),
    minNumber = cms.uint32(0)
)


