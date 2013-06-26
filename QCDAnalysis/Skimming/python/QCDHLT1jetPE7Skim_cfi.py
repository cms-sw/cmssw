import FWCore.ParameterSet.Config as cms

# 
# Module for Jet trigger skim -- HLT1jetPE7 trigger
#
QCDHLT1jetPE7Trigger = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1jetPE7'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)


