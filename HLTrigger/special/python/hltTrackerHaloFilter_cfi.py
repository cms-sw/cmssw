import FWCore.ParameterSet.Config as cms

hltTrackerHaloFilter = cms.EDFilter( "HLTTrackerHaloFilter",
   inputTag           = cms.InputTag( "hltSiStripClusters" ),
   saveTags = cms.bool( False ),
   MaxClustersTECp    = cms.int32(50),
   MaxClustersTECm    = cms.int32(50),
   SignalAccumulation = cms.int32(5),
   MaxClustersTEC     = cms.int32(60),
   MaxAccus           = cms.int32(4),
   FastProcessing     = cms.int32(1)
)


