import FWCore.ParameterSet.Config as cms

# module to filter on the maximal number of Jets
maxLayer1Jets = cms.EDFilter("PATJetMaxFilter",
    maxNumber = cms.uint32(999999),
    src = cms.InputTag("selectedLayer1Jets")
)


