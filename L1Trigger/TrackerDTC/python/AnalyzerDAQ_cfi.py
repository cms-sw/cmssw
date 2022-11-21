import FWCore.ParameterSet.Config as cms

TrackerDTCAnalyzerDAQ_params = cms.PSet (

  InputTagTTClusterDetSetVec = cms.InputTag( "TTClustersFromPhase2TrackerDigis", "ClusterInclusive" ) # original TTCluster selection

)
