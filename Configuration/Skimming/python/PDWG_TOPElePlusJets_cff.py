import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
TOPElePlusJets = hlt.triggerResultsFilter.clone()
TOPElePlusJets.triggerConditions = cms.vstring(
'HLT_Ele25_CaloIdVL_*',
'HLT_Ele25_CaloIdVT_*',
)
TOPElePlusJets.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
TOPElePlusJets.l1tResults = cms.InputTag("")
TOPElePlusJets.throw = cms.bool( False )
