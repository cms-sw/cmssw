import FWCore.ParameterSet.Config as cms


eventAuxiliaryHistoryProducer = cms.EDProducer("EventAuxiliaryHistoryProducer",
    historyDepth = cms.uint32(5)
)
