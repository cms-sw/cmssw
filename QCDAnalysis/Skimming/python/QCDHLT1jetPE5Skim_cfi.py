import FWCore.ParameterSet.Config as cms

# 
# Module for Jet trigger skim -- HLT1jetPE5 trigger
#
QCDHLT1jetPE5Trigger = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1jetPE5'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)


