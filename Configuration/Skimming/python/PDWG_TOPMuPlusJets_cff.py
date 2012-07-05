import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
TOPMuPlusJets = hlt.triggerResultsFilter.clone()
TOPMuPlusJets.triggerConditions = cms.vstring(
'HLT_IsoMu17_eta2p1_TriCentral*',
'HLT_IsoMu17_eta2p1_Central*',
'HLT_IsoMu17_eta2p1_DiCentral*',
'HLT_Mu17_eta2p1_Central*',
'HLT_Mu17_eta2p1_TriCentral*',
)
TOPMuPlusJets.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
TOPMuPlusJets.l1tResults = cms.InputTag("")
TOPMuPlusJets.throw = cms.bool( False )
