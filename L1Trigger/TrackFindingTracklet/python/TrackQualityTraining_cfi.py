# configuration for hybrid track reconstruction chain analyzer

import FWCore.ParameterSet.Config as cms

# This unit is relevant only when running the HYBRID_NEWKF Algorithm. # 

TrackQualityTraining_params = cms.PSet (

  TrainingMode              = cms.bool     ( False ), # By Default False.
  EvaluationMode            = cms.bool     ( False ), # By Default False.
  L1TrackInputTag           = cms.InputTag ("ProducerTFP", "TTTrackAccepted"),
  MCTruthTrackInputTag      = cms.InputTag ("TTTrackAssociatorFromPixelDigis", "TTTrackAccepted"),
  TTClusterTruth            = cms.InputTag ("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"),
  TTStubTruth               = cms.InputTag ("TTStubAssociatorFromPixelDigis", "StubAccepted"),

)
