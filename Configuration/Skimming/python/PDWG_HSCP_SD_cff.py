import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
HSCPSD = hlt.triggerResultsFilter.clone()
HSCPSD.triggerConditions = cms.vstring(
#    "HLT_StoppedHSCP*",
    "HLT_JetE*_NoBPTX_*",
    "HLT_JetE*_NoBPTX_NoHalo_*",
    "HLT_JetE*_NoBPTX3BX_NoHalo_*")
HSCPSD.hltResults = cms.InputTag( "TriggerResults", "", "HLT" )
HSCPSD.l1tResults = cms.InputTag("")
HSCPSD.throw = cms.bool( False )


