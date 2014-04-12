import FWCore.ParameterSet.Config as cms

# 
# Module for Jet trigger skim -- HLT1jetPE3 trigger
#
QCDHLT1jetPE3Trigger = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1jetPE3'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)


