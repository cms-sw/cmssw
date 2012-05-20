import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
upsilonHLT = hlt.triggerResultsFilter.clone()
upsilonHLT.triggerConditions = cms.vstring('HLT_Dimuon7_Upsilon*','HLT_Dimuon11_Upsilon*')
upsilonHLT.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
upsilonHLT.l1tResults = cms.InputTag("")
upsilonHLT.throw = cms.bool( False )
