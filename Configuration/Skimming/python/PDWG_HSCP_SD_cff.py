import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
HSCPSD = hlt.hltHighLevel.clone()
HSCPSD.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
HSCPSD.HLTPaths = cms.vstring(
    "HLT_StoppedHSCP*",
    "HLT_JetE*_NoBPTX*")
HSCPSD.andOr = cms.bool( True )
HSCPSD.throw = cms.bool( False )

