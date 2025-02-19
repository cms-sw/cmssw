import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
ZHbbSkim = hlt.hltHighLevel.clone()
ZHbbSkim.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
ZHbbSkim.HLTPaths = cms.vstring(
    "HLT_L1ETM40_v*",
    "HLT_DiCentralPFJet30_PFMET80_v*",
    "HLT_DiCentralPFJet30_PFMHT80_v*" )
ZHbbSkim.andOr = cms.bool( True )
ZHbbSkim.throw = cms.bool( False )

