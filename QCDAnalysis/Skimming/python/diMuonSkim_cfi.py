import FWCore.ParameterSet.Config as cms

#
#
# Modules for soft di-muon trigger skim.
#
diMuonTrigger = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT2MuonNonIso'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


