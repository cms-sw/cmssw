import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
TOPElePlusJets = hlt.triggerResultsFilter.clone()
TOPElePlusJets.triggerConditions = cms.vstring(
'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFJet30_BTagIPIter_v*',
'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFJet30_v*',
'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_DiCentralPFJet30_v*',
'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TriCentralPFJet30_v*',
'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TriCentralPFJet50_40_30_v*',
'HLT_Ele25_CaloIdVT_TrkIdT_TriCentralPFJet30_v*',
'HLT_Ele25_CaloIdVT_TrkIdT_TriCentralPFJet50_40_30_v*',
'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_BTagIPIter_v*',
'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFNoPUJet30_v*',
'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_DiCentralPFNoPUJet30_v*',
'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TriCentralPFNoPUJet30_v*',
'HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TriCentralPFNoPUJet50_40_30_v*',
'HLT_Ele25_CaloIdVT_TrkIdT_TriCentralPFNoPUJet30_v*',
'HLT_Ele25_CaloIdVT_TrkIdT_TriCentralPFNoPUJet50_40_30_v*',
)
TOPElePlusJets.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
TOPElePlusJets.l1tResults = cms.InputTag("")
TOPElePlusJets.throw = cms.bool( False )
