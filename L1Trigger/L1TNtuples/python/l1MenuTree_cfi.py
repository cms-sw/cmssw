import FWCore.ParameterSet.Config as cms

l1MenuTree = cms.EDAnalyzer("L1MenuTreeProducer",
   l1GtRecordInputTag = cms.InputTag("l1GtTriggerMenuLite"),
   l1GtReadoutRecordInputTag = cms.InputTag(""),
   l1GtTriggerMenuLiteInputTag =  cms.InputTag("l1GtTriggerMenuLite"),
)

