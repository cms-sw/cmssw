import FWCore.ParameterSet.Config as cms

#
#
# Modules for soft di-muon trigger skim.
#
diMuonTrigger = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_DoubleMu3'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


