import FWCore.ParameterSet.Config as cms

TrackerDTCAnalyzer_params = cms.PSet (

  #=== ED parameter

  ParamsAnalyzer = cms.PSet (
    ProducerLabel           = cms.string( "TrackerDTCProducer" ),                                     # label of DTC producer
    InputTagTTClusterAssMap = cms.InputTag( "TTClusterAssociatorFromPixelDigis", "ClusterAccepted" ), # tag of AssociationMap between TTCluster and TrackingParticles
    UseMCTruth              = cms.bool  ( True  )                                                     # open and analyze TrackingParticles, original TTStubs and Association between them
  ),

  #=== Cuts on MC truth particles (i.e., tracking particles) used for tracking efficiency measurements. And Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle)

  ParamsTP = cms.PSet (
    MinPt            = cms.double(  2.  ), # pt cut in GeV
    MaxEta           = cms.double(  2.4 ), # eta cut
    MaxVertR         = cms.double(  1.  ), # cut on vertex pos r in cm
    MaxVertZ         = cms.double( 30.  ), # cut on vertex pos z in cm
    MaxD0            = cms.double(  5.  ), # cut on impact parameter in cm
    MinLayers        = cms.int32 (  4   ), # required number of associated layers to a TP to consider it reconstruct-able
    MinLayersPS      = cms.int32 (  0   ), # required number of associated ps layers to a TP to consider it reconstruct-able
    MatchedLayers    = cms.int32 (  4   ), # required number of layers a found track has to have in common with a TP to consider it matched to it
    MatchedLayersPS  = cms.int32 (  0   ), # required number of ps layers a found track has to have in common with a TP to consider it matched to it
    UnMatchedStubs   = cms.int32 (  1   ), # allowed number of stubs a found track may have not in common with its matched TP
    UnMatchedStubsPS = cms.int32 (  0   )  # allowed number of PS stubs a found track may have not in common with its matched TP
  )

)
