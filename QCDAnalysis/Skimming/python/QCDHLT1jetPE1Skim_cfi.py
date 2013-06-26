import FWCore.ParameterSet.Config as cms

# 
# Module for Jet trigger skim -- HLT1jetPE1 trigger
#
QCDHLT1jetPE1Trigger = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1jetPE1'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)


