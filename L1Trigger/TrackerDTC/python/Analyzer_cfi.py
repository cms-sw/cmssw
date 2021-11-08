import FWCore.ParameterSet.Config as cms

TrackerDTCAnalyzer_params = cms.PSet (

  InputTagAccepted           = cms.InputTag( "TrackerDTCProducer",                "StubAccepted"     ), # dtc passed stubs selection
  InputTagLost               = cms.InputTag( "TrackerDTCProducer",                "StubLost"         ), # dtc lost stubs selection
  InputTagTTStubDetSetVec    = cms.InputTag( "TTStubsFromPhase2TrackerDigis",     "StubAccepted"     ), # original TTStub selection
  InputTagTTClusterDetSetVec = cms.InputTag( "TTClustersFromPhase2TrackerDigis",  "ClusterInclusive" ), # original TTCluster selection
  InputTagTTClusterAssMap    = cms.InputTag( "TTClusterAssociatorFromPixelDigis", "ClusterAccepted"  ), # tag of AssociationMap between TTCluster and TrackingParticles
  UseMCTruth                 = cms.bool( True )                                                         # eneables analyze of TPs                                                    # eneables analyze of TPs

)
