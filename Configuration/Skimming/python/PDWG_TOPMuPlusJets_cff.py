import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
TOPMuPlusJets = hlt.triggerResultsFilter.clone()
TOPMuPlusJets.triggerConditions = cms.vstring(
'HLT_IsoMu17_eta2p1_TriCentralPFJet30_v*',
'HLT_IsoMu20_eta2p1_CentralPFJet30_BTagIPIter_v*',
'HLT_IsoMu20_eta2p1_CentralPFJet30_v*',
'HLT_IsoMu20_eta2p1_DiCentralPFJet30_v*',
'HLT_IsoMu20_eta2p1_TriCentralPFJet30_v*',
'HLT_IsoMu20_eta2p1_TriCentralPFJet50_40_30_v*',
'HLT_Mu20_eta2p1_CentralPFJet30_BTagIPIter_v*',
'HLT_Mu20_eta2p1_TriCentralPFJet30_v*',
'HLT_Mu20_eta2p1_TriCentralPFJet50_40_30_v*',
'HLT_IsoMu20_eta2p1_CentralPFNoPUJet30_BTagIPIter_v*',
'HLT_IsoMu20_eta2p1_CentralPFNoPUJet30_v*',
'HLT_IsoMu20_eta2p1_DiCentralPFNoPUJet30_v*',
'HLT_IsoMu20_eta2p1_TriCentralPFNoPUJet30_v*',
'HLT_IsoMu20_eta2p1_TriCentralPFNoPUJet50_40_30_v*',
'HLT_Mu20_eta2p1_CentralPFNoPUJet30_BTagIPIter_v*',
'HLT_Mu20_eta2p1_TriCentralPFNoPUJet30_v*',
'HLT_Mu20_eta2p1_TriCentralPFNoPUJet50_40_30_v*',
)
TOPMuPlusJets.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
TOPMuPlusJets.l1tResults = cms.InputTag("")
TOPMuPlusJets.throw = cms.bool( False )
