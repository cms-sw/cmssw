import FWCore.ParameterSet.Config as cms


eventAuxiliaryHistoryProducer = cms.EDFilter("EventAuxiliaryHistoryProducer",
    historyDepth = cms.uint32(5)
)
