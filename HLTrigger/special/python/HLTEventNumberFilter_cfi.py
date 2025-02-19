import FWCore.ParameterSet.Config as cms

hltEventNumberFilter = cms.EDFilter( "HLTEventNumberFilter", 
   period = cms.uint32(4096), # accept if eventNumber%period ==0
   invert = cms.bool(True),   # if invert=true, invert this accept decision
   saveTags = cms.bool( False )
)
