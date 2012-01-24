import FWCore.ParameterSet.Config as cms

hltLogMonitorFilter = cms.EDFilter("HLTLogMonitorFilter",
    default_threshold = cms.uint32(10),
    categories = cms.VPSet( ),
    saveTags = cms.bool( False )
)
