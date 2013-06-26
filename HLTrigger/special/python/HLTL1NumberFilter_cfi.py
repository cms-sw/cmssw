import FWCore.ParameterSet.Config as cms

hltL1NumberFilter = cms.EDFilter( "HLTL1NumberFilter",
   rawInput = cms.InputTag("source"), 
   period = cms.uint32(4096), # accept if eventNumber%period ==0
   invert = cms.bool(True)    # if invert=true, invert this accept decision
)
